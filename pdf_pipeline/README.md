# pdf_pipeline

A PDF → Markdown extraction pipeline implementing this flow:

```
pdf
 └─ pdfplumber ──────────────┬──────────────────────────────┐
        │ words+bboxes(xyxy) │ image                        │
        ▼                    ▼                              │
  empty words?          pp-doclayout                         │
   yes │   no            (DocLayoutV3)                       │
       │     └────────────┐  │ layout_class+bbox+order       │
       ▼                  ▼  ▼                                
  OCR(pytesseract/   Plumber words + pp-doclayout matching    
  paddleocr/rapidocr)  (align_words_to_layout)                
       │                  │                                   
       ▼                  ▼                                   
  Matching ocr bboxes   (class_name, bbox, reading_order, text)
  and layout                       │
  (align_ocr_to_layout)            ▼
       │                   class_name == "table"?
       │                    yes │      no
       └───────────┬───────────┘       └─ formatted by class
                    ▼
            TableFormerONNX → OTSL → markdown
```

Both matching branches converge on the same `RegionText` shape and the same
downstream per-region rendering (`region_to_markdown`), including the
table → TableFormerONNX → OTSL → markdown branch — the diagram only draws
that branch coming off the pdfplumber-matching box, but a "table" layout
region can equally appear on a scanned (OCR'd) page, so it's handled the
same way in both paths.

## Modules

| File                  | Diagram piece(s) |
|------------------------|------------------|
| `layout.py`            | `pp-doclayout` (DocLayoutV3 ONNX wrapper) + both "matching" boxes (`align_words_to_layout`, `align_ocr_to_layout`, sharing the generic `align_tokens_to_layout` core) |
| `ocr_backends.py`      | `Ocr(pytesseract, paddleocr, rapidocr)` — also reused for table-cell OCR |
| `table_extraction.py`  | `TableFormerONNX` → OTSL → markdown |
| `pipeline.py`          | The `if the words are empty` branch, the `if classname is table` branch, and final markdown assembly (`process_pdf_page`, `process_pdf`) |

## What's new vs. the two source notebooks

The two original notebooks (`pdfplumber_.ipynb`, `docling_.ipynb`) already
had working code for: layout detection, the pdfplumber↔layout matcher, OCR
backends, and TableFormer→OTSL. What was missing, per the diagram, was:

- **The OCR-fallback matching box.** `align_words_to_layout` only knew how
  to consume pdfplumber's `extract_words()` output. `align_tokens_to_layout`
  generalizes the matching algorithm (containment + nearest-centroid
  fallback + line re-assembly) to take any `text + bbox` token list, in
  image-pixel space. `align_words_to_layout` and `align_ocr_to_layout` are
  now both adapters onto that one function.
- **The `if classname is table` branch wired into the page loop.** The
  notebook's per-region markdown loop only ever embedded a cropped image for
  `table`/`image`/`chart` regions; TableFormer was only run manually,
  separately, against one standalone table image. `region_to_markdown` now
  checks `class_name == "table"` and runs `table_image_to_otsl` +
  `otsl_to_markdown` automatically, falling back to an embedded image if no
  `TableFormerONNX` runner was supplied (or if extraction throws).
- **Table cells prefer pdfplumber's own words over OCR.** When the PDF has a
  real text layer, the words pdfplumber already extracted for a table region
  are reused directly as TableFormer's cell-text source (`tokens=` on
  `ocr_anchor_cells`/`table_image_to_otsl`) instead of re-OCR-ing the table
  crop — cheaper and avoids OCR mistakes on text that's already exact.
  `table_ocr_backend` is now only invoked as a fallback when a table region
  has no underlying words (e.g. the table is itself a scanned image, or the
  whole page is scanned and went through the page-level OCR branch).
- **Standalone image input.** `process_image_page`/`process_images` run the
  same diagram on plain images (.jpg/.png/etc. — a photographed or scanned
  page with no PDF structure at all), always via the OCR branch since
  there's no native text layer to check. `process_document` is a single
  dispatcher that picks the PDF path or the image path based on the input,
  so callers don't need to branch on file type themselves. Both share the
  same per-image core (`_process_rendered_page`) as the PDF path, so a
  scanned PDF page and a standalone photo of a page are handled identically
  once OCR kicks in.
- **Fixed a latent color-channel bug along the way.** `DocLayoutV3._preprocess`
  always applies a BGR→RGB swap, correct only if its input really is BGR
  (as `cv2.imread` produces). The original page-rendering code fed it a
  PIL-derived array (RGB) directly, which silently double-swapped the
  channels and degraded detection on every PDF page processed through this
  pipeline. Fixed via a shared `_to_bgr_array` helper used by both the PDF
  and image entry points.
- **`otsl_to_markdown`.** The OTSL→DocTagsDocument→DoclingDocument→markdown
  steps were ad hoc, later notebook cells, applied to one hardcoded `otsl`
  variable. Factored into one function so it can run once per detected
  table inside a page loop.

## Requirements

```
pip install pdfplumber numpy opencv-python onnxruntime pydantic pillow docling-core
# plus whichever OCR backend(s) you use:
pip install pytesseract            # needs the tesseract binary too
pip install rapidocr-onnxruntime
pip install paddleocr paddlepaddle
```

You'll also need the model artifacts referenced in the original notebooks:
- `PP-DocLayout/PP-DocLayoutV3.onnx`
- `tableformerv1/onnx/<variant>/tableformer_<variant>_{encoder,decoder_step,bbox_decoder}.onnx`
  + `tableformerv1/tm_config.json`

## Usage

```python
from pdf_pipeline import DocLayoutV3, TableFormerONNX, get_ocr_backend, process_document

layout_detector = DocLayoutV3("PP-DocLayout/PP-DocLayoutV3.onnx")
page_ocr_backend = get_ocr_backend("rapidocr")          # used on scanned PDF pages AND any standalone image
table_runner = TableFormerONNX(artifact_root="tableformerv1")
table_ocr_backend = get_ocr_backend("paddleocr")          # fallback for tables with no underlying text

# Works the same regardless of input type:
markdown_doc = process_document(
    "document.pdf",                       # or "scan.jpg", or ["page1.png", "page2.png"]
    layout_detector,
    page_ocr_backend=page_ocr_backend,
    table_runner=table_runner,
    table_ocr_backend=table_ocr_backend,
)

with open("document.md", "w") as f:
    f.write(markdown_doc)
```

`process_document` dispatches on what you pass it: a `.pdf` path goes
through `process_pdf` (pdfplumber, may have a native text layer); a single
image path (`.jpg`/`.png`/etc.) or a list of image paths goes through
`process_image_page`/`process_images` (always OCR, since standalone images
never have a text layer). Call `process_pdf`/`process_image_page` directly
if you want the type-specific kwargs (e.g. `pages=` or `resolution=` only
make sense for PDFs) or the full per-page debug dict instead of just the
markdown string.

For single-page debugging (mirrors the original notebook's main cell), use
`process_pdf_page(page, layout_detector, doc_path, ...)` directly — it
returns a dict with `markdown`, `regions`, `layout_result`, `image`, and
`used_ocr`, so you can inspect intermediate state (e.g. call
`layout_detector.visualize(result["layout_result"], "debug.png", result["image"])`)
before trusting the final markdown.

Cropped images (for `image`/`chart`/`table`-without-TableFormer regions) are
written to `<doc_stem>/images/<n>.png`, same convention as the original
notebook's `crop_and_save_image`.

## Notes / things to tune for your documents

- `containment_threshold` / `line_tol_ratio` (passed through `process_pdf_page`
  → `align_words_to_layout` / `align_ocr_to_layout` → `align_tokens_to_layout`)
  control how aggressively tokens are assigned to boxes and how lines are
  re-grouped; the defaults (0.5 / 0.5) come from the original notebook.
- `SKIP_CLASSES` and `IMAGE_LIKE_CLASSES` in `pipeline.py` control which
  layout classes are dropped vs. rendered as a markdown image link vs.
  rendered as text — extend these for your document types (e.g. treat
  `display_formula`/`inline_formula` specially instead of falling through
  to plain text).
- If `table_runner`/`table_ocr_backend` are omitted, table regions degrade
  gracefully to an embedded cropped image instead of raising.
