# Drawing QA Toolkit v2

AI-powered drawing review tool for **Autoliv / CET Thailand** — three
production-ready checks for the engineering drawing review workflow.

## What It Does

### 1. CET Revision Comparison
Compares a master drawing with a revision to verify that:
- All **requested changes** were correctly applied
- No **unintended/unauthorized changes** were made
- **Text and dimension** values changed as expected (OCR-based)

**How it works:**
- SIFT feature matching + RANSAC homography for automatic alignment
- SSIM structural similarity + pixel diff for discrepancy detection
- OCR extracts text → structured text-level diff catches subtle value changes
- Optional AI layer reconciles diffs against your change-request list

### 2. Prototype Completeness Check
For prototype requests, confirms that the drawing contains all required
manufacturing instructions:
- Base fabric type, sewing method, panel positioning
- Tether routing, inflator pocket, vent holes
- All items from a customizable YAML checklist

### 3. Typo & Cross-View Consistency Check
- **Typos**: Finds spelling mistakes using local OCR + spellcheck (domain-aware)
- **Mismatches**: Detects inconsistencies between views/sections (e.g., a
  dimension given as 120mm in the main view but 125mm in the section view)

---

## Installation (Windows — Full Local Setup)

### Step 1: Python 3.10+
You likely already have Python. Verify:
```powershell
python --version
```
If not installed, get it from https://www.python.org/downloads/

### Step 2: Tesseract OCR (System Install)
Tesseract is the primary OCR engine. It needs a separate system install:

**Option A — winget (recommended):**
```powershell
winget install UB-Mannheim.TesseractOCR
```

**Option B — Manual installer:**
1. Download from https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer
3. Install to the default path: `C:\Program Files\Tesseract-OCR`
4. **Add to PATH**: The installer usually does this, but verify by running:
   ```powershell
   tesseract --version
   ```
   If not found, add `C:\Program Files\Tesseract-OCR` to your system PATH.

### Step 3: Python Dependencies
```powershell
cd "path\to\Drawing"
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `opencv-python-headless` | Image processing, alignment, feature matching |
| `numpy` | Array operations |
| `pillow` | Image loading |
| `PyMuPDF` | PDF → image conversion |
| `pytesseract` | Tesseract OCR Python wrapper |
| `easyocr` | Deep learning OCR (handles rotated text better) |
| `pyspellchecker` | Local typo detection |
| `pyyaml` | YAML config loading |
| `scikit-image` | SSIM structural comparison |
| `openpyxl` | Excel report export |
| `anthropic` | Anthropic API client (optional, for cloud AI) |
| `ollama` | Ollama client (optional, for local AI) |

> **Note:** EasyOCR will download its model files (~100MB) on first run.
> This is a one-time download and works offline after that.

### Step 4: Local AI (Optional — for fully local operation)
If you want AI features to run 100% locally (no cloud API):

1. **Install Ollama**: Download from https://ollama.com/download/windows
2. **Pull a vision model**:
   ```powershell
   # Recommended: LLaVA 7B (~4.7GB download, needs ~8GB RAM)
   ollama pull llava:7b
   
   # Better quality but needs more RAM (~16GB):
   ollama pull llava:13b
   ```
3. Ollama runs as a background service — it starts automatically after install.

> **RAM Requirements:**
> - LLaVA 7B: ~8GB RAM
> - LLaVA 13B: ~16GB RAM
> - Text-only models (llama3, mistral): ~4-6GB RAM

### Step 5: Run
```powershell
streamlit run app.py
```
Opens in your browser at http://localhost:8501

---

## Project Structure

```
Drawing/
├── app.py                              # Streamlit UI (3 modes)
├── core/
│   ├── comparator.py                    # Mode 1: Enhanced CV diff (SIFT + SSIM)
│   ├── change_verifier.py               # Mode 1: AI reconciliation vs requested changes
│   ├── completeness_checker.py          # Mode 2: instruction checklist verification
│   ├── consistency_checker.py           # Mode 3: typo + cross-view mismatch detection
│   ├── pdf_handler.py                   # PDF → high-res images (PyMuPDF)
│   ├── ocr_engine.py                    # Local OCR (Tesseract + EasyOCR hybrid)
│   ├── text_analyzer.py                 # Local typo detection + dimension parsing
│   ├── structured_diff.py               # Text/dimension-level diff between drawings
│   └── local_llm.py                     # Ollama integration for local AI
├── config/
│   ├── autoliv_checklist.yaml           # Customizable prototype checklist
│   └── domain_terms.yaml                # Domain terms (won't be flagged as typos)
├── samples/
│   ├── master_drawing.png               # Sample master for Mode 1
│   ├── revised_drawing.png              # Sample revision for Mode 1
│   ├── prototype_drawing_sample.png     # Sample for Modes 2 & 3
│   ├── generate_sample_drawings.py
│   └── generate_prototype_sample.py
├── requirements.txt
└── README.md
```

---
**Bottom line:** The CV pipeline, OCR, text diff, and typo detection are 100% local
with no external dependencies. Only the "reasoning about what the change means" step
optionally uses an LLM (local via Ollama or cloud via Anthropic).

---

## Configuration

### Editing the Prototype Checklist
Edit `config/autoliv_checklist.yaml` — items are grouped by category:
```yaml
prototype_checklist:
  - "Base fabric type / material specification"
  - "Sewing method / stitch type"
  # ... add your items here
```

### Editing Domain Terms
Edit `config/domain_terms.yaml` — terms here won't be flagged as typos:
```yaml
material_terms:
  - "Nylon 420D"
  - "PU coated"
sewing_terms:
  - "lock-stitch"
  - "bartack"
```

### Switching AI Backend
In the sidebar, under "AI Backend":
- **Ollama (Local)** — select your installed vision model
- **Anthropic (Cloud)** — enter your API key

You can also set the API key as an environment variable:
```powershell
$env:ANTHROPIC_API_KEY = "your-key-here"
```

---

## Using Real Autoliv Drawings

1. Uncheck "Use sample drawings" in the sidebar
2. Upload your drawings (PDF, PNG, JPG, or TIFF)
3. For PDFs with multiple pages, select which page to compare
4. Tune `threshold` and `min_area` per drawing type:
   - Dense drawings with hatching/text: higher `min_area` (500+)
   - Clean CAD exports: lower values work well
5. Enable "OCR text diff" for text-level comparison
6. Edit the checklist and domain terms to match your specific drawing standards

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `tesseract not found` | Install Tesseract and add to PATH (see Step 2) |
| `easyocr` first run is slow | Normal — it downloads models (~100MB). One-time only. |
| SIFT not working | Your OpenCV may not include SIFT. Update: `pip install opencv-python-headless>=4.9` |
| Ollama connection refused | Make sure Ollama is running: `ollama serve` |
| PDF upload doesn't work | Install PyMuPDF: `pip install PyMuPDF` |
| Low alignment confidence | Drawings may be too different or heavily rotated. Try unchecking SIFT and using ORB. |
| Too many false positives | Increase `threshold` and `min_area` in the sidebar |
| Excel export missing | Install openpyxl: `pip install openpyxl` |
