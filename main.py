"""
PDF Pipeline — Streamlit app with markdown visualizer.

Run from the project root (next to the `pdf_pipeline/` package folder):

    streamlit run main.py
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download

from doc_pipeline import (
    DocLayoutV3,
    TableFormerONNX,
    get_ocr_backend,
    process_document,
    setup_pipeline_logging,
)

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

REC_KEYS_FILENAME = "ch_en_dict.txt"


# --------------------------------------------------------------------------- #
# Model path resolution — pulled out of module scope so it can be imported
# and unit-tested without triggering real downloads on `import main`.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelPaths:
    layout: str
    table_artifacts: str
    ocr_det_small: str
    ocr_rec_small: str
    ocr_det_medium: str
    ocr_rec_medium: str
    ocr_rec_keys: str


def _resolve_local_or_download(
    local_path: str,
    repo_id: str,
    local_dir: str,
    subfolder: str | None = None,
    filename: str = "inference.onnx",
    downloader=snapshot_download,
) -> str:
    """
    Return `local_path` if it already exists on disk; otherwise download the
    repo via `downloader` (defaults to `huggingface_hub.snapshot_download`,
    swappable in tests) and return the resolved path to `filename` inside it.

    Centralizing this "check local, else download" pattern also fixes three
    bugs present in the original per-model inline logic:
    - the medium OCR recognizer path used to skip appending `filename`,
      resolving to a directory instead of the actual model file
    - the small OCR recognizer's fallback downloaded into the *medium*
      model's local_dir by mistake
    - `snapshot_download` has no `subfolder` parameter (that belongs to
      `hf_hub_download`, a different function) — restricting the download
      to one folder of the repo requires `allow_patterns` instead, which is
      what's used here
    """
    if os.path.exists(local_path):
        return local_path
    kwargs = {"repo_id": repo_id, "local_dir": local_dir}
    if subfolder is not None:
        kwargs["allow_patterns"] = [f"{subfolder}/*"]
    downloaded_dir = downloader(**kwargs)
    return (
        os.path.join(downloaded_dir, subfolder, filename)
        if subfolder is not None
        else os.path.join(downloaded_dir, filename)
    )


def resolve_model_paths(downloader=snapshot_download) -> ModelPaths:
    """
    Resolve local paths to every model weight the pipeline needs, downloading
    only what's missing. Pure function of the local filesystem + `downloader`
    (injectable for tests — pass a fake/mock instead of hitting the network),
    so it can be exercised directly:

        paths = resolve_model_paths(downloader=fake_snapshot_download)
        assert paths.ocr_rec_medium.endswith("inference.onnx")

    Safe to call more than once — every branch just checks for an existing
    local file first.
    """
    layout = _resolve_local_or_download(
        "PP-DocLayout/inference.onnx",
        repo_id="PaddlePaddle/PP-DocLayoutV3_onnx",
        local_dir="PP-DocLayout",
        downloader=downloader,
    )

    table_artifacts = (
        "tableformerv1"
        if os.path.exists("tableformerv1")
        else downloader(repo_id="bakhil-aissa/tableformerv1", local_dir="tableformerv1")
    )

    ocr_det_medium = _resolve_local_or_download(
        "pp_ocr_medium/det/inference.onnx",
        repo_id="PaddlePaddle/PP-OCRv6_medium_det_onnx",
        local_dir="pp_ocr_medium",
        subfolder="det",
        downloader=downloader,
    )
    ocr_rec_medium = _resolve_local_or_download(
        "pp_ocr_medium/rec/inference.onnx",
        repo_id="PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        local_dir="pp_ocr_medium",
        subfolder="rec",
        downloader=downloader,
    )
    ocr_det_small = _resolve_local_or_download(
        "pp_ocr_small/det/inference.onnx",
        repo_id="PaddlePaddle/PP-OCRv6_small_det_onnx",
        local_dir="pp_ocr_small",
        subfolder="det",
        downloader=downloader,
    )
    ocr_rec_small = _resolve_local_or_download(
        "pp_ocr_small/rec/inference.onnx",
        repo_id="PaddlePaddle/PP-OCRv6_small_rec_onnx",
        local_dir="pp_ocr_small",  # fixed: was "pp_ocr_medium" in the original
        subfolder="rec",
        downloader=downloader,
    )

    return ModelPaths(
        layout=layout,
        table_artifacts=table_artifacts,
        ocr_det_small=ocr_det_small,
        ocr_rec_small=ocr_rec_small,
        ocr_det_medium=ocr_det_medium,
        ocr_rec_medium=ocr_rec_medium,
        ocr_rec_keys=REC_KEYS_FILENAME,
    )


# --------------------------------------------------------------------------- #
# Pipeline loading (Streamlit-cached; wraps the pure resolve step above)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models…")
def load_pipeline(
    layout_model: str,
    table_artifact_root: str,
    table_variant: str,
    ocr_backend_name: str,
    rec_path: str,
    det_path: str,
    rec_keys_path: str,
):
    setup_pipeline_logging(level="INFO")
    layout_detector = DocLayoutV3(layout_model)

    table_runner = TableFormerONNX(
        artifact_root=table_artifact_root,
        variant=table_variant,
    )
    if ocr_backend_name == "rapidocr":
        table_ocr_backend = get_ocr_backend(
            ocr_backend_name, det_model_path=det_path, rec_model_path=rec_path, rec_keys_path=rec_keys_path
        )
        page_ocr_backend = get_ocr_backend(
            ocr_backend_name, det_model_path=det_path, rec_model_path=rec_path, rec_keys_path=rec_keys_path
        )
    else:
        table_ocr_backend = get_ocr_backend(ocr_backend_name)
        page_ocr_backend = get_ocr_backend(ocr_backend_name)
    return layout_detector, page_ocr_backend, table_runner, table_ocr_backend


def parse_pages(raw: str) -> list[int] | None:
    raw = raw.strip()
    if not raw:
        return None
    pages: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        page = int(part)
        if page < 1:
            raise ValueError("Page numbers must be 1-based (1, 2, 3, …).")
        pages.append(page)
    return pages or None


def resolve_markdown_images(markdown: str, base_dir: Path) -> str:
    """Turn relative image links into absolute paths so Streamlit can render them."""

    def _replace(match: re.Match[str]) -> str:
        alt, path = match.group(1), match.group(2)
        if path.startswith(("http://", "https://", "data:")):
            return match.group(0)
        candidate = Path(path)
        if not candidate.is_file():
            candidate = (base_dir / path).resolve()
        if candidate.is_file():
            return f"![{alt}]({candidate.as_posix()})"
        return match.group(0)

    return IMAGE_LINK_RE.sub(_replace, markdown)


def save_upload(uploaded_file, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / uploaded_file.name
    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


def main() -> None:
    st.set_page_config(
        page_title="PDF Pipeline",
        page_icon="📄",
        layout="wide",
    )
    st.title("PDF Pipeline")
    st.caption("Extract structured markdown from PDFs and scanned images.")

    model_paths = resolve_model_paths()

    with st.sidebar:
        st.header("Settings")
        layout_model = st.selectbox(
            "Layout model",
            options=[model_paths.layout],
        )
        table_artifact_root = st.selectbox(
            "TableFormer artifacts",
            options=[model_paths.table_artifacts],
        )
        table_variant = st.selectbox(
            "TableFormer variant",
            options=["accurate"],
            index=0,
        )
        ocr_backend = st.selectbox(
            "OCR backend",
            options=["rapidocr", "pytesseract"],
            index=0,
        )
        if ocr_backend == "rapidocr":
            path_det = st.selectbox(
                "RapidOCR detector model",
                options=[model_paths.ocr_det_small, model_paths.ocr_det_medium],
                index=0,
            )
            path_rec = st.selectbox(
                "RapidOCR recognizer model",
                options=[model_paths.ocr_rec_small, model_paths.ocr_rec_medium],
                index=0,
            )
            path_keys = st.selectbox(
                "RapidOCR keys model",
                options=[model_paths.ocr_rec_keys],
                index=0,
            )

        resolution = st.slider("PDF render DPI", min_value=72, max_value=300, value=150)
        pages_raw = st.text_input(
            "PDF pages (optional)",
            placeholder="1, 2, 5 — leave empty for all pages",
        )

    uploaded = st.file_uploader(
        "Upload a PDF or image",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "gif"],
    )

    if uploaded is None:
        st.info("Upload a document to start.")
        return

    col_preview, col_meta = st.columns([2, 1], gap="large")
    with col_meta:
        st.markdown(f"**File:** `{uploaded.name}`")
        st.markdown(f"**Size:** {uploaded.size / 1024:.1f} KB")

    with col_preview:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix == ".pdf":
            pdf_b64 = base64.b64encode(uploaded.getvalue()).decode()
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{pdf_b64}" '
                'width="100%" height="640" style="border:none;"></iframe>',
                unsafe_allow_html=True,
            )
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
            st.image(uploaded, use_container_width=True)

    process = st.button("Extract markdown", type="primary", use_container_width=False)

    if not process:
        if "markdown_result" in st.session_state:
            markdown_doc = st.session_state["markdown_result"]
            doc_stem = st.session_state.get("doc_stem", PROJECT_ROOT)
        else:
            return
    else:
        try:
            pages = parse_pages(pages_raw) if pages_raw else None
        except ValueError as exc:
            st.error(str(exc))
            return

        layout_detector, page_ocr_backend, table_runner, table_ocr_backend = load_pipeline(
            layout_model,
            table_artifact_root,
            table_variant,
            ocr_backend,
            rec_path=path_rec,
            det_path=path_det,
            rec_keys_path=path_keys,
        )

        work_dir = PROJECT_ROOT / ".streamlit_output" / Path(uploaded.name).stem
        doc_path = save_upload(uploaded, work_dir)
        kwargs: dict = {"resolution": resolution}
        if pages is not None and doc_path.suffix.lower() == ".pdf":
            kwargs["pages"] = pages

        with st.spinner("Running pipeline…"):
            markdown_doc = process_document(
                str(doc_path),
                layout_detector,
                page_ocr_backend=page_ocr_backend,
                table_runner=table_runner,
                table_ocr_backend=table_ocr_backend,
                **kwargs,
            )

        st.session_state["markdown_result"] = markdown_doc
        st.session_state["doc_stem"] = work_dir
        st.session_state["output_name"] = doc_path.stem + ".md"
        st.success("Extraction complete.")

    preview_md = resolve_markdown_images(markdown_doc, Path(st.session_state.get("doc_stem", PROJECT_ROOT)))

    tab_preview, tab_source, tab_download = st.tabs(["Preview", "Markdown source", "Download"])

    with tab_preview:
        st.markdown(preview_md, unsafe_allow_html=False)

    with tab_source:
        st.code(markdown_doc, language="markdown")

    with tab_download:
        output_name = st.session_state.get("output_name", "output.md")
        st.download_button(
            label="Download .md file",
            data=markdown_doc,
            file_name=output_name,
            mime="text/markdown",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()