# Drawing QA Toolkit — Demo

Three prototype checks, built as one app with a mode switcher in the
sidebar. All three ship with synthetic sample drawings so you can demo
before you have real Autoliv/CET data.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pick a mode in the sidebar:

### 1. CET Revision Comparison
Deterministic pixel-diff (OpenCV) finds every changed region between a
master drawing and a revision, then an optional AI layer reconciles those
regions against a list of changes you actually requested — sorting them
into **applied**, **unintended** (changed but not requested), and
**missing** (requested but not found). This is the answer to *"verify
requested modifications were applied and no unnecessary changes were made."*

### 2. Prototype Completeness Check
For prototype requests: reads a single drawing (all views + notes) and
checks it against an editable checklist — default items are base fabric
type, sewing method, and panel positioning method. Flags each item as
**present** (with the evidence text found), **missing**, or **ambiguous**.
The sample drawing is missing a panel-positioning instruction on purpose.

### 3. Typo & Cross-View Consistency Check
Reads all text on a drawing sheet and:
- flags genuine spelling mistakes (skips an editable domain-terms allow-list
  so real jargon/material names aren't misflagged)
- flags cases where the same panel/feature/dimension is described
  differently between the main view and a section/detail view

The sample sheet has a planted typo ("seem allowance") and a planted
mismatch (panel width 120mm in the main view vs 125mm in the section view).

## How it maps to your original pipeline diagram (Mode 1 CV layer)

| Slide stage | File | What it does |
|---|---|---|
| Master Drawing / New Revision | `app.py` (upload or sample) | Loads the two images |
| Pre-processing (grayscale + normalization) | `core/comparator.py: preprocess()` | Grayscale, contrast normalization, denoise blur |
| Image Alignment (OpenCV feature matching) | `align_images()` | ORB keypoints + BFMatcher + RANSAC homography — corrects scan skew/offset/rotation |
| Pixel Subtraction (NumPy matrix diff) | `compute_discrepancy_map()` | `cv2.absdiff` between aligned grayscale images |
| Threshold Filter (noise removal) | same function | Binary threshold + morphological open/close/dilate |
| Discrepancy Map (highlighted regions) | `extract_discrepancies()` + `annotate()` | Contour detection → bounding boxes, severity-graded, color-coded overlay |
| Report Output (flagged changes exported) | `ComparisonResult.to_report_dict()` | Structured JSON, downloadable from the UI |

Modes 2 and 3 are a different kind of problem — single-drawing information
extraction and cross-reference reasoning rather than pixel diffing — so they
skip the CV pipeline entirely and go straight to a vision-LLM call per
drawing. See `core/completeness_checker.py` and `core/consistency_checker.py`.

## Project structure

```
drawing_comparator/
├── app.py                            # Streamlit UI (3 modes)
├── core/
│   ├── comparator.py                  # Mode 1: deterministic CV diff pipeline
│   ├── change_verifier.py             # Mode 1: AI reconciliation vs requested changes
│   ├── completeness_checker.py        # Mode 2: instruction checklist verification
│   └── consistency_checker.py         # Mode 3: typo + cross-view mismatch detection
├── samples/
│   ├── generate_sample_drawings.py    # Master/revision pair for Mode 1
│   └── generate_prototype_sample.py   # Multi-view sheet for Modes 2 & 3
├── requirements.txt
└── README.md
```

## What all three AI-based checks have in common

Each is one Claude vision API call per drawing/region, asking for a strict
JSON response, which the code parses into a typed result. I tested the
plumbing (image crop/encode → prompt → JSON parse → aggregation) against
mocked API responses, since I don't have your API key — the code paths work,
but prompt *quality* against real drawing content needs validation once you
run it against actual CET drawings. Expect to tune the prompts in each
`core/*.py` file after seeing real output; drawing conventions and
terminology are specific to your standard and the model won't know them
without examples.

Cost/privacy note: each check is a small number of API calls with image
inputs — cheap per drawing. If Autoliv drawings are sensitive, look into
Anthropic's Zero Data Retention agreements before sending real production
drawings through any API-based tool.

## Change-Request Verification (AI) — Mode 1 detail

1. Type or paste your requested changes into the sidebar (one per line) —
   this is whatever you'd normally put in the ECO note or email to CET.
2. Enter your Anthropic API key (get one at console.anthropic.com — this
   demo calls the API directly, it does not go through claude.ai).
3. Run the comparison. For every region the CV pipeline flags as changed,
   it crops a before/after pair and asks Claude's vision model whether that
   specific region matches one of your requested changes.
4. You get three buckets:
   - **Applied as requested** — matched, confirmed done
   - **Unintended changes** — real geometric changes CET made that nobody
     asked for
   - **Requested but missing** — things you asked for that don't show up in
     any detected region

## What existing commercial tools do (researched for feature parity)

Tools like Bluebeam Studio compare, DraftSight Draw Compare, Trimble's
drawing overlay, and CoLab's revision comparison all converge on the same
core UX, which this demo replicates:

- **Auto-alignment first** — never compare raw pixels without correcting
  scan offset/rotation, or you get false positives everywhere.
- **Color-coded severity**, not just a flat diff mask.
- **A clickable list of discrepancies**, not just an image — engineers
  triage a list, they don't hunt across a picture.
- **Exportable report** for the ECO/QA record.

Two things production tools add that this demo does **not** yet include,
worth mentioning if this goes further:
- **Vector-native comparison** (reading DWG/DXF entities directly) for CAD
  exports — pixel diffing only works reliably on rasterized/scanned input.
- **OCR-based text/dimension comparison** — catches cases where a dimension
  value changes but the geometry pixel-shift is too small to threshold
  cleanly (e.g., "25.4" → "25.5"). Worth adding via `pytesseract` once you
  have a target format.

## Swapping in real Autoliv drawings later

1. Uncheck "Use sample drawings" in the sidebar and upload two images
   directly — no code changes needed for PNG/JPG scans or exports.
2. If your drawings are PDFs, convert pages to images first
   (`pdf2image` or `PyMuPDF`) — I can wire that in when you're ready.
3. Tune `threshold` and `min_area` in the sidebar per drawing type — dense
   drawings with lots of hatching/text need a higher `min_area` to avoid
   noise; clean CAD exports can go lower.
