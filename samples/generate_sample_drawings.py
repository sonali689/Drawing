"""
Generates a synthetic pair of engineering drawings (master + revised) so the
comparator can be demoed before real company drawings are available.

Known differences injected into the revision (for validation/demo purposes):
  1. A circle (bolt hole) moved position
  2. A dimension text value changed (25.4 -> 28.0)
  3. A slot/rectangle removed
  4. A new fillet/circle added
  5. A centerline shifted slightly (simulates scan skew, should be ignored by
     alignment, NOT flagged as a real diff)
"""
import cv2
import numpy as np
import os

W, H = 1400, 1000
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def base_drawing(shift=(0, 0), rotate_deg=0.0):
    """Draws a simple mechanical part (front view + dimensions + title block)."""
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    dx, dy = shift

    # Outer plate
    cv2.rectangle(img, (150 + dx, 150 + dy), (1000 + dx, 700 + dy), (0, 0, 0), 3)

    # Bolt holes (4 corners)
    holes = [(220, 220), (930, 220), (220, 630), (930, 630)]
    for (hx, hy) in holes:
        cv2.circle(img, (hx + dx, hy + dy), 20, (0, 0, 0), 2)

    # Center bore
    cv2.circle(img, (575 + dx, 425 + dy), 80, (0, 0, 0), 3)
    cv2.circle(img, (575 + dx, 425 + dy), 40, (0, 0, 0), 2)

    # Keyway slot
    cv2.rectangle(img, (555 + dx, 250 + dy), (595 + dx, 300 + dy), (0, 0, 0), 2)

    # Centerlines (dashed)
    for x in range(150, 1000, 20):
        cv2.line(img, (x + dx, 425 + dy), (x + 10 + dx, 425 + dy), (0, 0, 0), 1)
    for y in range(150, 700, 20):
        cv2.line(img, (575 + dx, y + dy), (575 + dx, y + 10 + dy), (0, 0, 0), 1)

    # Dimension lines + text
    cv2.line(img, (150 + dx, 730 + dy), (1000 + dx, 730 + dy), (0, 0, 0), 1)
    cv2.putText(img, "850.0", (540 + dx, 760 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(img, "R25.4", (600 + dx, 415 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Title block
    cv2.rectangle(img, (1000 + dx, 850 + dy), (1350 + dx, 950 + dy), (0, 0, 0), 2)
    cv2.putText(img, "PART: BRACKET-104", (1010 + dx, 880 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "REV: A", (1010 + dx, 905 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "SCALE 1:1", (1010 + dx, 930 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if rotate_deg != 0:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rotate_deg, 1.0)
        img = cv2.warpAffine(img, M, (W, H), borderValue=(255, 255, 255))

    return img


def make_master():
    return base_drawing(shift=(0, 0), rotate_deg=0.0)


def make_revision():
    # Simulate a slightly skewed/shifted scan (should be corrected by alignment,
    # not reported as a real difference)
    img = base_drawing(shift=(6, -4), rotate_deg=0.4)

    # --- Real design changes injected on top ---
    # 1. Move top-right bolt hole
    cv2.circle(img, (930 + 6, 220 - 4), 20, (255, 255, 255), -1)  # erase old
    cv2.circle(img, (960 + 6, 250 - 4), 20, (0, 0, 0), 2)         # draw new

    # 2. Change dimension text 25.4 -> 28.0 (paint over + rewrite)
    cv2.rectangle(img, (595, 395), (680, 425), (255, 255, 255), -1)
    cv2.putText(img, "R28.0", (600 + 6, 415 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 3. Remove keyway slot
    cv2.rectangle(img, (550, 245), (600, 305), (255, 255, 255), -1)

    # 4. Add a new small hole (center-left)
    cv2.circle(img, (350 + 6, 425 - 4), 15, (0, 0, 0), 2)

    # 5. Bump revision letter
    cv2.rectangle(img, (1075, 895), (1160, 915), (255, 255, 255), -1)
    cv2.putText(img, "REV: B", (1010 + 6, 905 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


if __name__ == "__main__":
    master = make_master()
    revision = make_revision()
    cv2.imwrite(os.path.join(OUT_DIR, "master_drawing.png"), master)
    cv2.imwrite(os.path.join(OUT_DIR, "revised_drawing.png"), revision)
    print("Saved master_drawing.png and revised_drawing.png to", OUT_DIR)
