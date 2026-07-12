"""
table_extraction.py
====================

Right-hand branch of the diagram: when a matched region's class is "table",
its cropped image goes through TableFormerONNX -> OTSL string -> markdown.

`TableFormerONNX`, `ocr_anchor_cells`, `seq_to_otsl`, and `table_image_to_otsl`
are carried over from docling_.ipynb essentially unchanged (decoupled from
that notebook's specific OCR-backend import style; backends now come from
ocr_backends.py). `otsl_to_markdown` is new — it's the "otsl -> markdown" step
in the diagram, which in the original notebook was done ad hoc in later
cells (build a <doctag>, run DocTagsDocument/DoclingDocument). Folding it
into one function makes it reusable per-table inside the full-page pipeline.
It returns `(markdown, dataframe)` — the dataframe is built from
`TableItem.export_to_dataframe()` with `collapse_spanned_columns` applied,
since a raw rectangular DataFrame otherwise duplicates a colspan'd cell's
value into every column it visually spans (see that function's docstring
for why, and why `pd.read_html(export_to_html())` has the identical issue).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from .ocr_backends import OCRBackend

logger = logging.getLogger(__name__)

ANCHOR_TAGS = {"fcel", "ecel", "ched", "rhed", "srow"}  # tags that occupy their own grid slot
EMPTY_OTSL_PLACEHOLDER = "?"  # inserted when a non-empty-tagged cell OCRs to nothing,
                              # so the docling_core parser never sees an empty content cell

MEAN = np.array([0.94247851, 0.94254675, 0.94292611], dtype=np.float32)
STD = np.array([0.17910956, 0.17940403, 0.17931663], dtype=np.float32)
IMAGE_SIZE = 448


# --------------------------------------------------------------------------- #
# bbox merge helpers (lcel chains)
# --------------------------------------------------------------------------- #
def merge_bboxes(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Union the cxcywh boxes of an lcel chain's stub anchor and the real cell that closes it."""
    new_w = (box2[0] + box2[2] / 2) - (box1[0] - box1[2] / 2)
    new_h = (box2[1] + box2[3] / 2) - (box1[1] - box1[3] / 2)
    new_left = box1[0] - box1[2] / 2
    new_top = min((box2[1] - box2[3] / 2), (box1[1] - box1[3] / 2))
    return np.array(
        [new_left + new_w / 2, new_top + new_h / 2, new_w, new_h], dtype=np.float32
    )


def apply_bbox_merge(coords: np.ndarray, bboxes_to_merge: dict) -> np.ndarray:
    coords = [np.asarray(c, dtype=np.float32) for c in coords]
    merged, skip = [], set()
    for i, box1 in enumerate(coords):
        if i in bboxes_to_merge:
            j = bboxes_to_merge[i]
            if j >= 0:
                skip.add(j)
                merged.append(merge_bboxes(box1, coords[j]))
        elif i not in skip:
            merged.append(box1)
    return np.stack(merged) if merged else np.empty((0, 4), dtype=np.float32)


def make_session(path: str, threads: int = 2) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_mem_reuse = True
    opts.enable_profiling = False
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------- #
# TableFormer ONNX runner
# --------------------------------------------------------------------------- #
class TableFormerONNX:
    """
    encoder + decoder-step (autoregressive loop, KV-cache) + bbox_decoder,
    all ONNX Runtime. Produces a structure sequence (OTSL tags) plus a
    normalized cxcywh bbox per anchor cell.

    Usage:
        runner = TableFormerONNX(artifact_root="tableformerv1", variant="accurate")
    """

    def __init__(self, artifact_root: str = "tableformerv1", variant: str = "accurate", threads: int = 2):
        root = Path(artifact_root)
        onnx_dir = root / "onnx" / variant
        self.enc = make_session(str(onnx_dir / f"tableformer_{variant}_encoder.onnx"), threads)
        self.dec = make_session(str(onnx_dir / f"tableformer_{variant}_decoder_step.onnx"), threads)
        self.bbox = make_session(str(onnx_dir / f"tableformer_{variant}_bbox_decoder.onnx"), threads)

        with open(root / "tm_config.json", encoding="utf-8") as f:
            self.tag_map = json.load(f)["dataset_wordmap"]["word_map_tag"]
        self.rev_tag = {v: k for k, v in self.tag_map.items()}

        cache_input = next(i for i in self.dec.get_inputs() if i.name == "cache")
        self.num_layers = int(cache_input.shape[0])
        self.embed_dim = int(cache_input.shape[3])

    @staticmethod
    def is_valid_url(url_string: str) -> bool:
        try:
            result = urlparse(url_string)
            return all([result.scheme in ("http", "https"), result.netloc])
        except ValueError:
            return False

    def image_preprocess(self, img) -> np.ndarray:
        """Accepts a path, URL, PIL.Image, or HWC RGB ndarray."""
        if isinstance(img, str) and self.is_valid_url(img):
            import requests

            img = Image.open(requests.get(img, stream=True, timeout=30).raw)
        if isinstance(img, (str, Path)):
            img = Image.open(img)
        if isinstance(img, Image.Image):
            img = np.array(img.convert("RGB"))
        elif isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        img = (img.astype(np.float32) - 255.0 * MEAN) / STD
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        img = img.transpose(2, 1, 0) / 255.0
        return np.expand_dims(img.astype(np.float32), axis=0)

    def draw_bbox(self, img_path: str, bboxs: np.ndarray, out_path: str = "draw_p.png") -> None:
        """Debug helper — draws predicted cxcywh boxes on the ORIGINAL-resolution image."""
        from PIL import ImageDraw

        pil_image = Image.open(img_path)
        width, height = pil_image.size
        draw = ImageDraw.Draw(pil_image)
        for cx, cy, w, h in np.asarray(bboxs):
            xyxy = (
                int((cx - 0.5 * w) * width), int((cy - 0.5 * h) * height),
                int((cx + 0.5 * w) * width), int((cy + 0.5 * h) * height),
            )
            draw.rectangle(xyxy, outline="red", width=2)
        pil_image.save(out_path)

    def predict(self, img: np.ndarray, max_steps: int = 1024) -> dict:
        """img: float32 [1, 3, 448, 448] (preprocessed). Returns seq / outputs_class /
        outputs_coord (cxcywh, normalized [0,1]) / bboxes_to_merge / timings."""
        t0 = time.time()
        enc_out, memory = self.enc.run(None, {"image": img})
        t_enc = time.time() - t0

        wm = self.tag_map
        decoded_tags = np.array([[wm["<start>"]]], dtype=np.int64)
        cache = np.zeros((self.num_layers, 0, 1, self.embed_dim), dtype=np.float32)
        skip_next_tag = True
        prev_tag_ucel = False
        line_num = 0
        first_lcel = True
        bbox_ind = 0
        cur_bbox_ind = -1
        bboxes_to_merge: dict = {}
        tag_H_buf = []

        t1 = time.time()
        for _ in range(max_steps):
            logits, last_hidden, cache = self.dec.run(
                None, {"decoded_tags": decoded_tags, "memory": memory, "cache": cache}
            )
            new_tag = int(np.argmax(logits, axis=1)[0])

            if line_num == 0 and new_tag == wm["xcel"]:
                new_tag = wm["lcel"]
            if prev_tag_ucel and new_tag == wm["lcel"]:
                new_tag = wm["fcel"]

            if new_tag == wm["<end>"]:
                decoded_tags = np.concatenate(
                    [decoded_tags, np.array([[new_tag]], dtype=np.int64)], axis=0
                )
                break

            if not skip_next_tag:
                if new_tag in (wm["fcel"], wm["ecel"], wm["ched"], wm["rhed"],
                               wm["srow"], wm["nl"], wm["ucel"]):
                    tag_H_buf.append(last_hidden[:, 0, :].copy())
                    if first_lcel is not True:
                        bboxes_to_merge[cur_bbox_ind] = bbox_ind
                    bbox_ind += 1

            if new_tag != wm["lcel"]:
                first_lcel = True
            else:
                if first_lcel:
                    tag_H_buf.append(last_hidden[:, 0, :].copy())
                    first_lcel = False
                    cur_bbox_ind = bbox_ind
                    bboxes_to_merge[cur_bbox_ind] = -1
                    bbox_ind += 1

            skip_next_tag = new_tag in (wm["nl"], wm["ucel"], wm["xcel"])
            prev_tag_ucel = new_tag == wm["ucel"]
            if new_tag == wm["nl"]:
                line_num += 1

            decoded_tags = np.concatenate(
                [decoded_tags, np.array([[new_tag]], dtype=np.int64)], axis=0
            )
        t_dec = time.time() - t1

        seq = decoded_tags.squeeze().tolist()

        if tag_H_buf:
            tag_H_stacked = np.stack(
                [h[None, ...] if h.ndim == 1 else h for h in tag_H_buf], axis=0
            ).reshape(-1, 1, self.embed_dim).astype(np.float32)
            t2 = time.time()
            cls_logits, coord = self.bbox.run(None, {"enc_out": enc_out, "tag_H_stacked": tag_H_stacked})
            t_bbox = time.time() - t2
            coord = apply_bbox_merge(coord, bboxes_to_merge)
        else:
            cls_logits = np.empty((0, 3), dtype=np.float32)
            coord = np.empty((0, 4), dtype=np.float32)
            t_bbox = 0.0

        return {
            "seq": seq,
            "outputs_class": cls_logits,
            "outputs_coord": coord,
            "bboxes_to_merge": bboxes_to_merge,
            "timings": {"encoder": t_enc, "decoder": t_dec, "bbox": t_bbox,
                        "total": t_enc + t_dec + t_bbox},
        }


# --------------------------------------------------------------------------- #
# OTSL assembly: structure tokens + per-cell OCR -> docling-ready OTSL string
# --------------------------------------------------------------------------- #
def compute_ioa(ocr_bbox: list[float], cell_bbox: list[float]) -> float:
    """Intersection over OCR-Token Area (IoA): maps a token to a cell if most
    of the token sits inside it."""
    ox1, oy1, ox2, oy2 = ocr_bbox
    cx1, cy1, cx2, cy2 = cell_bbox

    ix1 = max(ox1, cx1)
    iy1 = max(oy1, cy1)
    ix2 = min(ox2, cx2)
    iy2 = min(oy2, cy2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    ocr_area = (ox2 - ox1) * (oy2 - oy1)
    return inter_area / max(ocr_area, 1e-6)


def ocr_anchor_cells(
    orig_image_path: str,
    seq: list,
    box_o: np.ndarray,
    rev_tag: dict,
    backend: Optional[OCRBackend] = None,
    tokens: Optional[list[dict]] = None,
    pad: int = 4,
    narrow_width: float = 0.08,
) -> list:
    """Maps text tokens into TableFormer cell boundaries via spatial intersection.

    Either `backend` (run OCR on the table crop) or pre-extracted `tokens`
    (e.g. pdfplumber words already expressed in the crop's local pixel space —
    see pipeline.py) must be supplied. `tokens` takes priority: when a PDF
    has a real text layer, reusing those exact words is both cheaper and more
    accurate than re-OCRing a table crop pdfplumber already read correctly.
    """
    if tokens is None and backend is None:
        raise ValueError("ocr_anchor_cells needs either `tokens` or `backend`")

    orig = Image.open(orig_image_path).convert("RGB")
    orig_w, orig_h = orig.size

    anchor_tags = [rev_tag[t] for t in seq[1:-1] if rev_tag[t] in ANCHOR_TAGS]
    assert len(anchor_tags) == box_o.shape[0], (
        f"{len(anchor_tags)} anchor tags vs {box_o.shape[0]} boxes — structural alignment issue."
    )

    ocr_tokens = tokens if tokens is not None else backend.get_text_boxes(orig)
    source = "pdfplumber tokens" if tokens is not None else type(backend).__name__
    logger.debug(
        "ocr_anchor_cells: %d anchor cells, %d tokens from %s",
        len(anchor_tags),
        len(ocr_tokens),
        source,
    )
    texts = []

    for i, (tag, (cx, cy, w, h)) in enumerate(zip(anchor_tags, box_o)):
        if tag == "ecel":
            texts.append("")
            continue

        x1 = max(0, (cx - w / 2) * orig_w - pad)
        y1 = max(0, (cy - h / 2) * orig_h - pad)
        x2 = min(orig_w, (cx + w / 2) * orig_w + pad)
        y2 = min(orig_h, (cy + h / 2) * orig_h + pad)
        cell_bbox = [x1, y1, x2, y2]

        matched_tokens = [t for t in ocr_tokens if compute_ioa(t["bbox"], cell_bbox) >= 0.45]

        if not matched_tokens:
            texts.append("")
            logger.debug("  cell[%d] tag=%s (empty)", i, tag)
            continue

        matched_tokens.sort(key=lambda t: (int(t["bbox"][1] / 6), t["bbox"][0]))
        cell_string = " ".join(t["text"] for t in matched_tokens)
        texts.append(cell_string.strip())
        logger.debug("  cell[%d] tag=%s text=%r", i, tag, cell_string.strip()[:60])

    return texts


def _otsl_cell_text(tag: str, text: str) -> str | None:
    if tag == "ecel":
        return None
    cleaned = (text or "").strip()
    return cleaned if cleaned else EMPTY_OTSL_PLACEHOLDER


def seq_to_otsl(seq: list, rev_tag: dict, texts: list) -> str:
    """Re-walk seq, inserting OCR text right after each anchor tag."""
    out, ti = [], 0
    for tok in seq[1:-1]:
        name = rev_tag[tok]
        if name == "nl":
            out.append("<nl>")
            continue
        out.append(f"<{name}>")
        if name in ANCHOR_TAGS:
            cell_text = _otsl_cell_text(name, texts[ti])
            if cell_text is not None:
                out.append(cell_text)
            ti += 1
    return "".join(out)


def table_image_to_otsl(
    runner: TableFormerONNX,
    orig_image_path: str,
    backend: Optional[OCRBackend] = None,
    tokens: Optional[list[dict]] = None,
) -> str:
    """End-to-end: image -> ONNX TableFormer -> anchors filled from `tokens`
    (preferred, e.g. pdfplumber words) or OCR'd via `backend` -> OTSL string."""
    logger.info("TableFormer: processing %s", orig_image_path)
    preproc = runner.image_preprocess(orig_image_path)
    out = runner.predict(preproc)
    logger.debug(
        "TableFormer: seq length=%d, anchor boxes=%d",
        len(out["seq"]),
        out["outputs_coord"].shape[0],
    )
    texts = ocr_anchor_cells(
        orig_image_path, out["seq"], out["outputs_coord"], runner.rev_tag,
        backend=backend, tokens=tokens,
    )
    otsl = seq_to_otsl(out["seq"], runner.rev_tag, texts)
    logger.debug("TableFormer OTSL preview: %s", otsl[:200])
    return otsl


def collapse_spanned_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Collapse adjacent columns that are exact duplicates (same header text,
    identical value in every row) back into a single column.

    A plain rectangular DataFrame has no native way to represent a cell that
    spans multiple columns (e.g. an OTSL `lcel` chain / HTML `colspan`).
    Both `TableItem.export_to_dataframe()` and the `export_to_html()` +
    `pd.read_html()` round-trip handle this the same way: they repeat the
    spanned cell's value into every column it covers, so a 2-column-wide
    "Description" header ends up as two separate "Description" columns
    with identical content. This collapses those back into one.

    Note: this is a heuristic — if two genuinely distinct adjacent columns
    happen to share both the same header text and identical values in
    every row of this specific table, they'll also get collapsed. That's
    expected to be rare in practice (real tables don't usually have two
    different columns with the same name and identical data), but worth
    knowing if a table doesn't collapse cleanly.
    """
    cols = list(df.columns)
    if len(cols) <= 1:
        return df

    keep = [0]
    for i in range(1, len(cols)):
        prev = keep[-1]
        # pandas suffixes repeated header names with ".1", ".2", ... on
        # read_html; export_to_dataframe doesn't, so strip defensively.
        prev_header = str(cols[prev]).split(".")[0]
        cur_header = str(cols[i]).split(".")[0]
        same_header = prev_header == cur_header
        same_values = df.iloc[:, prev].astype(str).equals(df.iloc[:, i].astype(str))
        if same_header and same_values:
            continue  # duplicate of the column we're keeping — drop it
        keep.append(i)

    out = df.iloc[:, keep].copy()
    out.columns = [str(c).split(".")[0] for c in out.columns]
    return out


def _parse_md_table_row(line: str) -> list[str]:
    """Split one markdown table row into its cell strings, stripping whitespace."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    """Return True if every cell is a markdown alignment/separator marker (e.g. `---`, `:---:`)."""
    return all(set(c.replace(":", "").replace("-", "")) == set() and len(c) > 0 for c in cells)


def collapse_markdown_table(md: str) -> str:
    """
    Apply the same duplicate-adjacent-column collapse to every markdown table
    inside `md`, in-place. This is necessary because markdown has no colspan
    syntax — docling_core's `export_to_markdown()` represents a colspan-N cell
    by repeating its value in N consecutive columns, which looks like
    `| Description | Description | Total |` for a 2-column-wide "Description"
    header. This function detects and removes those duplicate columns from the
    markdown string so the rendered table matches what the source document
    actually meant.

    The collapse criterion is identical to `collapse_spanned_columns`: two
    adjacent columns are merged when they share the same header text (after
    stripping pandas `.1`/`.2` dedup suffixes, though docling markdown doesn't
    add those) AND every data row has the same value in both columns.

    Non-table lines (headings, paragraphs, etc.) are passed through unchanged.
    """
    lines = md.splitlines()
    out_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect the start of a markdown table: a line containing at least one |
        # that isn't just punctuation.
        if "|" not in line:
            out_lines.append(line)
            i += 1
            continue

        # Collect the full table block (contiguous lines containing |).
        table_block: list[str] = []
        while i < len(lines) and "|" in lines[i]:
            table_block.append(lines[i])
            i += 1

        if len(table_block) < 2:
            # Not a real table (need at least header + separator).
            out_lines.extend(table_block)
            continue

        # Parse into header, optional separator, and data rows.
        header_cells = _parse_md_table_row(table_block[0])
        sep_idx = 1 if _is_separator_row(_parse_md_table_row(table_block[1])) else None

        data_rows: list[list[str]] = []
        for row_line in table_block[(2 if sep_idx is not None else 1):]:
            data_rows.append(_parse_md_table_row(row_line))

        if not data_rows:
            out_lines.extend(table_block)
            continue

        # Pad short rows so indexing is safe.
        n_cols = len(header_cells)
        data_rows = [r + [""] * max(0, n_cols - len(r)) for r in data_rows]

        # Determine which columns to keep — same logic as collapse_spanned_columns.
        keep: list[int] = [0]
        for j in range(1, n_cols):
            prev = keep[-1]
            prev_header = header_cells[prev].split(".")[0]
            cur_header = header_cells[j].split(".")[0]
            same_header = prev_header == cur_header
            same_values = all(
                (row[prev] if prev < len(row) else "") == (row[j] if j < len(row) else "")
                for row in data_rows
            )
            if same_header and same_values:
                continue
            keep.append(j)

        if len(keep) == n_cols:
            # Nothing to collapse — pass through unchanged.
            out_lines.extend(table_block)
            continue

        # Rebuild the table with only the kept columns.
        def _fmt_row(cells: list[str], indices: list[int]) -> str:
            return "| " + " | ".join(cells[k] for k in indices) + " |"

        def _fmt_sep(indices: list[int]) -> str:
            return "| " + " | ".join("---" for _ in indices) + " |"

        out_lines.append(_fmt_row(header_cells, keep))
        if sep_idx is not None:
            out_lines.append(_fmt_sep(keep))
        for row in data_rows:
            out_lines.append(_fmt_row(row, keep))

    return "\n".join(out_lines)


def otsl_to_markdown(
    otsl: str, dummy_image_size: tuple[int, int] = (512, 512)
) -> tuple[str, "pd.DataFrame"]:
    """
    Wrap a raw OTSL tag string as <doctag><otsl>...</otsl></doctag>, load it
    through docling_core's DocTagsDocument/DoclingDocument, and export both
    markdown and a DataFrame for the (first) detected table.

    Returns `(markdown, dataframe)` — both have duplicate columns from
    colspan cells collapsed:
    - markdown: via `collapse_markdown_table` (markdown has no colspan syntax
      so docling repeats the spanned cell's text into every column it covers)
    - dataframe: via `TableItem.export_to_dataframe()` + `collapse_spanned_columns`
      (same duplication issue at the DataFrame level, same fix)
    """
    from docling_core.types.doc.document import DocTagsDocument, DoclingDocument

    otsl_doc = f"<doctag><otsl>{otsl}</otsl></doctag>"
    dummy_image = Image.new("RGB", dummy_image_size, color="black")
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([otsl_doc], [dummy_image])
    doc = DoclingDocument.load_from_doctags(doctags_doc)

    md = collapse_markdown_table(doc.export_to_markdown())

    if doc.tables:
        df = doc.tables[0].export_to_dataframe(doc=doc)
        df = collapse_spanned_columns(df)
        logger.info("OTSL -> markdown: %d rows x %d cols", df.shape[0], df.shape[1])
    else:
        df = pd.DataFrame()
        logger.warning("OTSL -> markdown: no tables detected in OTSL output")

    return md, df
