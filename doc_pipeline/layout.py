"""
layout.py
=========

Two responsibilities, matching the top half of the pipeline diagram:

1. `DocLayoutV3` — thin ONNXRuntime wrapper around PP-DocLayoutV3. Given a
   page image it returns layout boxes: (class_name, bbox[xyxy], reading_order).
   This is unchanged from the original pdfplumber_.ipynb.

2. Matching: turn a flat list of "tokens" (word/line-level text + bbox) into
   per-layout-region text, by containment + nearest-centroid fallback, then
   re-assembling lines within each region.

   The original notebook only had `align_words_to_layout`, which is hard-wired
   to pdfplumber's `page.extract_words()` output (point-space, needs scaling
   to image pixels). The diagram has a *second*, parallel matching box
   ("Matching ocr bboxes and layout") for the branch where pdfplumber finds no
   extractable words and an OCR backend is used instead. OCR tokens are
   already in image-pixel space (OCR runs on the same rendered image that was
   fed to PP-DocLayout), so no scaling is needed there.

   Rather than duplicating the matching logic, `align_tokens_to_layout` is the
   single generic implementation. `align_words_to_layout` and
   `align_ocr_to_layout` are thin adapters that normalize their respective
   inputs into the same token shape and call it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

import cv2
import numpy as np
from pydantic import BaseModel, Field

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - allows importing this module for
    ort = None        # static inspection without onnxruntime installed.


LAYOUT_CLASSES = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
]


# --------------------------------------------------------------------------- #
# Layout detection result types
# --------------------------------------------------------------------------- #
class BBox(BaseModel):
    """Axis-aligned bounding box in original image pixel coordinates."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


class LayoutBox(BaseModel):
    """A single detected layout element."""

    class_id: int
    class_name: str
    score: float
    bbox: BBox
    reading_order: int

    model_config = {"frozen": True}


class DocLayoutResult(BaseModel):
    """Full result for one image."""

    image_path: Optional[str] = None
    image_width: int
    image_height: int
    boxes: List[LayoutBox] = Field(default_factory=list)

    def sorted_by_reading_order(self) -> List[LayoutBox]:
        return sorted(self.boxes, key=lambda b: b.reading_order)


# --------------------------------------------------------------------------- #
# PP-DocLayoutV3 ONNX wrapper
# --------------------------------------------------------------------------- #

class DocLayoutV3:
    """
    Thin OO wrapper around the PP-DocLayoutV3 ONNX model.

    Usage:
        detector = DocLayoutV3("PP-DocLayout/PP-DocLayoutV3.onnx")
        result = detector.predict("page.png")
        for box in result.sorted_by_reading_order():
            print(box.class_name, box.bbox, box.reading_order)

        # optional: draw + save a visualization
        detector.visualize(result, "pp_doclayout.png")
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        target_input_size: Tuple[int, int] = (800, 800),
        score_threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
        enable_mem_pattern: bool = True,
        graph_optimization_level: "ort.GraphOptimizationLevel" = None,
    ):
        if ort is None:
            raise ImportError("onnxruntime is required for DocLayoutV3")

        graph_optimization_level = (
            graph_optimization_level or ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.model_path = str(model_path)
        self.target_input_size = target_input_size
        self.score_threshold = score_threshold
        self.class_names = class_names or LAYOUT_CLASSES

        self.session = self._build_session(
            providers=providers,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
            enable_mem_pattern=enable_mem_pattern,
            graph_optimization_level=graph_optimization_level,
        )
        self.providers_used = self.session.get_providers()

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self._mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._std =     std = np.array([1.0, 1.0, 1.0], dtype=np.float32)


    # ------------------------------------------------------------------ #
    def _build_session(
        self,
        providers: Optional[List[str]],
        intra_op_num_threads: Optional[int],
        inter_op_num_threads: Optional[int],
        enable_mem_pattern: bool,
        graph_optimization_level,
    ) -> "ort.InferenceSession":
        available = ort.get_available_providers()

        if providers is not None:
            resolved = [p for p in providers if p in available]
            if not resolved:
                raise ValueError(
                    f"None of the requested providers {providers} are "
                    f"available. Available providers: {available}"
                )
        else:
            preferred_order = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            resolved = [p for p in preferred_order if p in available]
            if not resolved:
                resolved = available or ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = graph_optimization_level
        sess_options.enable_mem_pattern = enable_mem_pattern
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if intra_op_num_threads is not None:
            sess_options.intra_op_num_threads = intra_op_num_threads
        if inter_op_num_threads is not None:
            sess_options.inter_op_num_threads = inter_op_num_threads

        session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=resolved,
        )

        actual = session.get_providers()
        if resolved[0] not in actual:
            import warnings

            warnings.warn(
                f"Requested provider order {resolved} but session is "
                f"actually running on {actual}.",
                RuntimeWarning,
                stacklevel=2,
            )

        return session
    
    #--------------------------------------------------------------------#
    def add_border_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Add a border to the image to ensure it is square.
        """
        orig_h, orig_w = image.shape[:2]
        if orig_h > orig_w:
            pad_h = (orig_h - orig_w) // 2
            pad_v = (orig_h - orig_w) // 2
        else:
            pad_h = (orig_w - orig_h) // 2
            pad_v = (orig_w - orig_h) // 2
        return cv2.copyMakeBorder(
            image,
            pad_v,
            pad_v,
            pad_h,
            pad_h,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        height, width = image.shape[:2]
        max_dim = max(height, width)
        border = (max_dim - height) // 2, (max_dim - width) // 2
        image = cv2.copyMakeBorder(
            image,
            border[0],
            border[0],  
            border[1],
            border[1],
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
        return image    
    # ------------------------------------------------------------------ #
    def median_filter(self, image:np.ndarray):
        """
        Apply median filter to the image.
        """
        return cv2.medianBlur(image, 3)
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.target_input_size
        #image = self.add_border_to_image(image)
        

        scale_h = target_h / orig_h
        scale_w = target_w / orig_w

        
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = (blob - self._mean) / self._std

        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return blob, scale_h, scale_w

    def _infer(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        orig_h, orig_w = image.shape[:2]
        input_blob, scale_h, scale_w = self._preprocess(image)

        target_h, target_w = self.target_input_size
        preprocess_shape = [np.array([target_h, target_w], dtype=np.float32)]

        input_feed = {
            self.input_names[0]: preprocess_shape,
            self.input_names[1]: input_blob,
            self.input_names[2]: [[scale_h, scale_w]],
        }

        # shape=(N, 7): [label_index, score, xmin, ymin, xmax, ymax, read_order]
        output = self.session.run(self.output_names, input_feed)[0]
        return output, orig_h, orig_w

    def _to_result(
        self,
        raw_boxes: np.ndarray,
        orig_h: int,
        orig_w: int,
        image_path: Optional[str],
    ) -> DocLayoutResult:
        logger.info(f"raw_boxes: {raw_boxes}")
        
        filtered = raw_boxes[raw_boxes[:, 1] > self.score_threshold]
        filtered = filtered[np.argsort(filtered[:, 5])]
        

        boxes: List[LayoutBox] = []
        for row in filtered:
            cls_id = int(row[0])
            boxes.append(
                LayoutBox(
                    class_id=cls_id,
                    class_name=self.class_names[cls_id]
                    if 0 <= cls_id < len(self.class_names)
                    else str(cls_id),
                    score=float(row[1]),
                    bbox=BBox(
                        xmin=float(row[2]),
                        ymin=float(row[3]),
                        xmax=float(row[4]),
                        ymax=float(row[5]),
                    ),
                    reading_order=int(row[6]),
                )
            )

        result = DocLayoutResult(
            image_path=image_path,
            image_width=orig_w,
            image_height=orig_h,
            boxes=boxes,
        )
        logger.info(
            "DocLayoutV3: %dx%d image -> %d boxes (threshold=%.2f)",
            orig_w,
            orig_h,
            len(boxes),
            self.score_threshold,
        )
        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict(self, image: Union[str, Path, np.ndarray]) -> DocLayoutResult:
        image_path: Optional[str] = None
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_array = cv2.imread(image_path)
            if img_array is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
        else:
            img_array = image

        raw_boxes, orig_h, orig_w = self._infer(img_array)
        return self._to_result(raw_boxes, orig_h, orig_w, image_path)

    def predict_batch(
        self, images: List[Union[str, Path, np.ndarray]]
    ) -> List[DocLayoutResult]:
        return [self.predict(img) for img in images]

    def visualize(
        self,
        result: DocLayoutResult,
        output_path: Union[str, Path],
        source_image: Optional[Union[str, Path, np.ndarray]] = None,
    ) -> str:
        src = source_image if source_image is not None else result.image_path
        if src is None:
            raise ValueError(
                "No source image available to draw on; pass source_image=..."
            )

        if isinstance(src, (str, Path)):
            img = cv2.imread(str(src))
            if img is None:
                raise FileNotFoundError(f"Could not read image: {src}")
        else:
            img = src.copy()

        for box in result.boxes:
            x0, y0, x1, y1 = (int(round(v)) for v in box.bbox.as_tuple())
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            label = f"{box.reading_order}|{box.class_name}"
            text_y = max(y0 - 10, 0)
            cv2.putText(
                img,
                label,
                (x0, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        output_path = str(output_path)
        cv2.imwrite(output_path, img)
        return output_path


def log_layout_result(result: DocLayoutResult) -> None:
    """Log every detected layout box at INFO level."""
    for box in result.sorted_by_reading_order():
        b = box.bbox
        logger.info(
            "  layout [%d] %s score=%.3f bbox=(%.0f, %.0f, %.0f, %.0f)",
            box.reading_order,
            box.class_name,
            box.score,
            b.xmin,
            b.ymin,
            b.xmax,
            b.ymax,
        )


def print_layout_result(result: DocLayoutResult) -> None:
    """Print layout boxes to stdout (legacy helper; prefer ``log_layout_result``)."""
    log_layout_result(result)


# --------------------------------------------------------------------------- #
# Token <-> layout matching (the two "matching" boxes in the diagram)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlacedWord:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float


@dataclass
class RegionText:
    reading_order: int
    class_name: str
    bbox: tuple  # (xmin, ymin, xmax, ymax) in image-pixel space
    text: str
    words: List[PlacedWord]


def _line_cluster_fast(words: List[PlacedWord], tol: float) -> List[List[PlacedWord]]:
    """Group words into lines by `top` proximity, then sort each line left-to-right.
    `tol` is computed once globally by the caller instead of per-region."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: w.top)
    lines: List[List[PlacedWord]] = []
    current = [ordered[0]]
    current_top = ordered[0].top
    for w in ordered[1:]:
        if abs(w.top - current_top) <= tol:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
            current_top = w.top
    lines.append(current)
    for line in lines:
        line.sort(key=lambda w: w.x0)
    return lines


def align_tokens_to_layout(
    tokens: Sequence[PlacedWord],
    layout_result: DocLayoutResult,
    containment_threshold: float = 0.5,
    line_tol_ratio: float = 0.5,
) -> List[RegionText]:
    """
    Generic matcher: given tokens already expressed in image-pixel space
    (same coordinate system as `layout_result`), assign each token to its
    best-matching layout box (smallest box containing it above
    `containment_threshold`, falling back to nearest centroid for tokens
    that don't sit inside any box), then re-assemble lines of text per box.

    This is the shared core behind both `align_words_to_layout` (pdfplumber
    path) and `align_ocr_to_layout` (OCR fallback path) in the diagram.
    """
    boxes = layout_result.boxes
    if not boxes or not tokens:
        logger.debug(
            "align_tokens_to_layout: skipped (boxes=%d, tokens=%d)",
            len(boxes) if boxes else 0,
            len(tokens) if tokens else 0,
        )
        return []

    logger.debug(
        "align_tokens_to_layout: %d tokens -> %d layout boxes (containment=%.2f)",
        len(tokens),
        len(boxes),
        containment_threshold,
    )

    wx0 = np.fromiter((t.x0 for t in tokens), dtype=np.float64, count=len(tokens))
    wx1 = np.fromiter((t.x1 for t in tokens), dtype=np.float64, count=len(tokens))
    wtop = np.fromiter((t.top for t in tokens), dtype=np.float64, count=len(tokens))
    wbot = np.fromiter((t.bottom for t in tokens), dtype=np.float64, count=len(tokens))
    word_area = np.clip((wx1 - wx0) * (wbot - wtop), 1e-6, None)

    bx0 = np.array([b.bbox.xmin for b in boxes])
    by0 = np.array([b.bbox.ymin for b in boxes])
    bx1 = np.array([b.bbox.xmax for b in boxes])
    by1 = np.array([b.bbox.ymax for b in boxes])
    box_area = np.clip((bx1 - bx0) * (by1 - by0), 1e-6, None)

    ix0 = np.maximum(wx0[:, None], bx0[None, :])
    iy0 = np.maximum(wtop[:, None], by0[None, :])
    ix1 = np.minimum(wx1[:, None], bx1[None, :])
    iy1 = np.minimum(wbot[:, None], by1[None, :])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    containment = inter / word_area[:, None]

    masked_area = np.where(containment >= containment_threshold, box_area[None, :], np.inf)
    best = np.argmin(masked_area, axis=1)
    matched = np.isfinite(masked_area[np.arange(len(tokens)), best])
    if not matched.all():
        bcx, bcy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        wcx, wcy = (wx0 + wx1) / 2.0, (wtop + wbot) / 2.0
        dist = (wcx[:, None] - bcx[None, :]) ** 2 + (wcy[:, None] - bcy[None, :]) ** 2
        best = np.where(matched, best, np.argmin(dist, axis=1))

    global_tol = float(np.median(wbot - wtop)) * line_tol_ratio

    order = np.argsort(best, kind="stable")
    sorted_idx = best[order]
    split_points = np.searchsorted(sorted_idx, np.arange(len(boxes)))
    split_points = np.append(split_points, len(order))

    results = []
    for box_idx, box in enumerate(boxes):
        idxs = order[split_points[box_idx]:split_points[box_idx + 1]]
        box_words = [tokens[i] for i in idxs]
        lines = _line_cluster_fast(box_words, global_tol)
        text = "\n".join(" ".join(w.text for w in line) for line in lines)
        results.append(
            RegionText(box.reading_order, box.class_name, box.bbox.as_tuple(), text, box_words)
        )
        logger.debug(
            "  matched box [%d] %s: %d words, text_len=%d",
            box.reading_order,
            box.class_name,
            len(box_words),
            len(text),
        )
    results.sort(key=lambda r: r.reading_order)
    return results


def pdfplumber_tokens_in_image_space(
    page, image_width: int, image_height: int
) -> List[PlacedWord]:
    """
    Extract a pdfplumber page's words and scale them from PDF point-space
    into the pixel space of a page image rendered at
    (image_width, image_height) — e.g. via `page.to_image(resolution=...)`,
    the same image PP-DocLayout ran on.

    Returns `[]` if `extract_words()` finds nothing (a scanned page with no
    text layer). Factored out of `align_words_to_layout` so callers that need
    these same scaled tokens for something else (e.g. pulling exact words for
    a table region instead of OCR-ing it) don't have to re-derive them.
    """
    raw_words = page.extract_words(
        x_tolerance=2,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
    )
    if not raw_words:
        logger.debug("pdfplumber extract_words: 0 words (scanned or image-only page)")
        return []

    scale_x = image_width / page.width
    scale_y = image_height / page.height
    logger.debug(
        "pdfplumber extract_words: %d words scaled to %dx%d (scale %.3f, %.3f)",
        len(raw_words),
        image_width,
        image_height,
        scale_x,
        scale_y,
    )
    return [
        PlacedWord(
            text=w["text"],
            x0=w["x0"] * scale_x,
            x1=w["x1"] * scale_x,
            top=w["top"] * scale_y,
            bottom=w["bottom"] * scale_y,
        )
        for w in raw_words
    ]


def align_words_to_layout(
    layout_result: DocLayoutResult,
    page,
    containment_threshold: float = 0.5,
    line_tol_ratio: float = 0.5,
) -> List[RegionText]:
    """
    Plumber-path matcher ("Plumber_bboxes_words and pp-doclayout matching" box).

    `page` is a pdfplumber Page. Its `extract_words()` output is in PDF point
    space, so it's scaled up to the layout image's pixel space before matching.
    """
    if not layout_result.boxes:
        return []

    tokens = pdfplumber_tokens_in_image_space(
        page, layout_result.image_width, layout_result.image_height
    )
    if not tokens:
        return []
    return align_tokens_to_layout(tokens, layout_result, containment_threshold, line_tol_ratio)


def align_ocr_to_layout(
    layout_result: DocLayoutResult,
    ocr_tokens: Sequence[Dict],
    containment_threshold: float = 0.5,
    line_tol_ratio: float = 0.5,
) -> List[RegionText]:
    """
    OCR-fallback matcher ("Matching ocr bboxes and layout" box) — used when
    pdfplumber's `extract_words()` returns nothing (e.g. a scanned page).

    `ocr_tokens` is the output of an `OCRBackend.get_text_boxes(image)` call
    (see ocr_backends.py): a list of {"text": str, "bbox": [x1, y1, x2, y2]}
    dicts already in the rendered image's pixel space — the same space the
    layout image was detected in — so no scaling is required here.
    """
    if not layout_result.boxes or not ocr_tokens:
        logger.debug(
            "align_ocr_to_layout: skipped (boxes=%d, ocr_tokens=%d)",
            len(layout_result.boxes),
            len(ocr_tokens),
        )
        return []

    logger.info(
        "OCR matching: %d OCR tokens -> %d layout boxes",
        len(ocr_tokens),
        len(layout_result.boxes),
    )

    tokens = [
        PlacedWord(
            text=t["text"],
            x0=float(t["bbox"][0]),
            x1=float(t["bbox"][2]),
            top=float(t["bbox"][1]),
            bottom=float(t["bbox"][3]),
        )
        for t in ocr_tokens
        if t.get("text")
    ]
    return align_tokens_to_layout(tokens, layout_result, containment_threshold, line_tol_ratio)
