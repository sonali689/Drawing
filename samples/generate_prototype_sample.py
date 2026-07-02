"""
Generates a synthetic prototype drawing sheet with TWO views (main + section)
and deliberately planted issues, so the completeness checker and the
typo/cross-view consistency checker have something real to catch:

  1. MISSING instruction: no "PANEL POSITIONING METHOD" note anywhere
     (present: BASE FABRIC, SEWING METHOD -- absent: panel positioning)
  2. TYPO: "SEEM ALLOWANCE" instead of "SEAM ALLOWANCE" in the section view
  3. CROSS-VIEW MISMATCH: main view calls the side panel width "120mm",
     the section view calls the same panel "125mm"
"""
import cv2
import numpy as np
import os

W, H = 1600, 1000
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_prototype_sheet():
    img = np.ones((H, W, 3), dtype=np.uint8) * 255

    # --- MAIN VIEW (left) ---
    cv2.putText(img, "MAIN VIEW", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.rectangle(img, (80, 100), (650, 550), (0, 0, 0), 2)
    # side panel
    cv2.rectangle(img, (80, 100), (200, 550), (0, 0, 0), 2)
    cv2.putText(img, "SIDE PANEL W=120mm", (85, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Notes block for main view
    cv2.putText(img, "BASE FABRIC: Nylon 420D, PU coated", (80, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(img, "SEWING METHOD: Lock-stitch, 6mm seam allowance", (80, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(img, "COLOR: Black, Pantone 19-4007", (80, 720), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    # NOTE: no panel positioning method note here (intentionally missing)

    # --- SECTION VIEW (right) ---
    cv2.putText(img, "SECTION A-A", (900, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.rectangle(img, (900, 100), (1500, 400), (0, 0, 0), 2)
    cv2.line(img, (900, 250), (1500, 250), (0, 0, 0), 1)
    cv2.putText(img, "SIDE PANEL W=125mm", (905, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # mismatch vs 120mm
    cv2.putText(img, "STITCH DETAIL: double row, 6mm seem allowance", (900, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # typo: "seem"

    # Title block
    cv2.rectangle(img, (1250, 850), (1550, 950), (0, 0, 0), 2)
    cv2.putText(img, "PART: PROTOTYPE-CET-221", (1260, 880), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "REQ TYPE: PROTOTYPE", (1260, 905), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "DATE: 2026-07-02", (1260, 930), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


if __name__ == "__main__":
    img = make_prototype_sheet()
    path = os.path.join(OUT_DIR, "prototype_drawing_sample.png")
    cv2.imwrite(path, img)
    print("Saved", path)
