import glob
import os
import cv2
import numpy as np

def score_ai_screenshot(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    b, g, r = cv2.split(img.astype(np.float32))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. VAE Decoder Chroma-Bleed Metric (Neutral pixels where R, G, B should be nearly equal)
    max_ch = np.maximum(np.maximum(r, g), b)
    min_ch = np.minimum(np.minimum(r, g), b)
    chroma_spread = max_ch - min_ch

    # Neutral UI regions (white cards, black backgrounds, dark grey bubbles) have spread <= 12
    neutral_mask = chroma_spread <= 12.0
    # In native screenshots, neutral pixels have spread == 0 (or exact uniform JPEG block)
    # In AI VAE-decoded images, neutral pixels have tiny 1..6 chroma jitter
    jitter_pixels = (chroma_spread >= 1.0) & (chroma_spread <= 8.0)
    vae_chroma_jitter = float(np.count_nonzero(jitter_pixels)) / float(max(np.count_nonzero(neutral_mask), 1))

    # 2. High-Frequency Edge vs Diffusion Blur Ratio (Laplacian 99th percentile vs 50th percentile)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    edge_crispness = float(np.percentile(lap, 99.5)) / (float(np.mean(lap)) + 1e-5)

    # 3. Bottom-Right Corner Watermark Signal
    roi = gray[int(h * 0.90):int(h * 0.99), int(w * 0.87):int(w * 0.985)].astype(np.float32)
    wm_diff = np.abs(roi - np.median(roi)) if roi.size > 0 else np.array([0.0])
    wm_ratio = float(np.count_nonzero(wm_diff >= 5.5)) / float(max(roi.size, 1))
    has_wm = (0.04 <= wm_ratio <= 0.38) and (float(np.max(wm_diff)) >= 10.0)

    # 4. 8x8 JPEG Block Boundary Ratio
    gf = gray.astype(np.float32)
    dx = np.abs(gf[:, 1:] - gf[:, :-1])
    dy = np.abs(gf[1:, :] - gf[:-1, :])
    bx, ix = dx[:, 7::8], np.delete(dx, np.arange(7, dx.shape[1], 8), axis=1)
    by, iy = dy[7::8, :], np.delete(dy, np.arange(7, dy.shape[0], 8), axis=0)
    grid_ratio = ((float(np.mean(bx)) + float(np.mean(by))) / 2.0) / (((float(np.mean(ix)) + float(np.mean(iy))) / 2.0) + 1e-5)

    return {
        "file": os.path.basename(filepath),
        "vae_jitter": round(vae_chroma_jitter, 4),
        "crispness": round(edge_crispness, 2),
        "grid_ratio": round(grid_ratio, 3),
        "has_wm": has_wm
    }

print("=== AUTHENTIC ===")
for p in glob.glob("dataset/authentic/**/*.*", recursive=True):
    if os.path.isfile(p):
        print(score_ai_screenshot(p))

print("\n=== EDITED ===")
for p in glob.glob("dataset/edited/**/*.*", recursive=True):
    if os.path.isfile(p):
        print(score_ai_screenshot(p))