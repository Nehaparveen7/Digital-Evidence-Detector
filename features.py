import os
import cv2
import numpy as np
from PIL import Image


def extract_forensic_features(filepath):
    """Extracts a 26-feature forensic vector tuned for AI-edited screenshots."""
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    b, g_ch, r = cv2.split(img.astype(np.float32))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 1. File & Aspect Features
    ext_is_png = 1.0 if os.path.splitext(filepath)[1].lower() == ".png" else 0.0
    aspect_ratio = float(w) / float(max(h, 1))

    # 2. Metadata Signals
    has_ai_meta = 0.0
    has_os_tag = 0.0
    try:
        with Image.open(filepath) as im:
            for k, v in im.info.items():
                txt = f"{k} {v}".lower()
                if any(m in txt for m in ["google", "gemini", "imagen", "synthid", "xmp", "adobe", "photoshop", "canva"]):
                    has_ai_meta = 1.0
                if any(s in txt for s in ["screenshot", "android", "iphone", "ios"]):
                    has_os_tag = 1.0
    except Exception:
        pass

    # 3. VAE Decoder Chroma-Bleed Metric (Key discriminator across all apps)
    max_ch = np.maximum(np.maximum(r, g_ch), b)
    min_ch = np.minimum(np.minimum(r, g_ch), b)
    chroma_spread = max_ch - min_ch
    neutral_mask = chroma_spread <= 12.0
    jitter_pixels = (chroma_spread >= 1.0) & (chroma_spread <= 8.0)
    vae_jitter = float(np.count_nonzero(jitter_pixels)) / float(max(np.count_nonzero(neutral_mask), 1))
    mean_neutral_spread = float(np.mean(chroma_spread[neutral_mask])) if np.any(neutral_mask) else 0.0

    # 4. Edge Crispness (Laplacian 99.5th percentile / mean)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    edge_crispness = float(np.percentile(lap, 99.5)) / (float(np.mean(lap)) + 1e-5)

    # 5. ELA Residual Statistics (Quality 90)
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    ela = cv2.cvtColor(cv2.absdiff(img, dec), cv2.COLOR_BGR2GRAY).astype(np.float32)
    ela_mean = float(np.mean(ela))
    ela_std = float(np.std(ela))
    ela_p95 = float(np.percentile(ela, 95))

    # 6. 8x8 JPEG Block Grid Ratio
    gf = gray.astype(np.float32)
    diff_x = np.abs(gf[:, 1:] - gf[:, :-1])
    diff_y = np.abs(gf[1:, :] - gf[:-1, :])
    bx = diff_x[:, 7::8]
    ix = np.delete(diff_x, np.arange(7, diff_x.shape[1], 8), axis=1)
    by = diff_y[7::8, :]
    iy = np.delete(diff_y, np.arange(7, diff_y.shape[0], 8), axis=0)
    grid_ratio = ((float(np.mean(bx)) + float(np.mean(by))) / 2.0) / (
        ((float(np.mean(ix)) + float(np.mean(iy))) / 2.0) + 1e-5
    )

    # 7. High-Frequency Median Noise Residual
    med = cv2.medianBlur(gray, 3)
    res = cv2.absdiff(gray, med).astype(np.float32)
    res_mean = float(np.mean(res))
    res_std = float(np.std(res))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    flat_mask = grad < 6.0
    flat_res_mean = float(np.mean(res[flat_mask])) if np.any(flat_mask) else res_mean
    flat_ela_mean = float(np.mean(ela[flat_mask])) if np.any(flat_mask) else ela_mean

    # 8. 2D FFT High-Frequency Spectrum Energy
    resized = cv2.resize(gray, (256, 256)).astype(np.float32)
    f_shift = np.fft.fftshift(np.fft.fft2(resized))
    mag = np.log1p(np.abs(f_shift))
    cy, cx = 128, 128
    y_idx, x_idx = np.ogrid[:256, :256]
    dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    mid_freq = float(np.mean(mag[(dist >= 20) & (dist < 65)]))
    high_freq = float(np.mean(mag[dist >= 65]))
    fft_ratio = high_freq / (mid_freq + 1e-5)

    # 9. LAB & HSV Color Shift Statistics
    a_mean = float(np.mean(lab[:, :, 1]))
    a_std = float(np.std(lab[:, :, 1]))
    b_mean = float(np.mean(lab[:, :, 2]))
    b_std = float(np.std(lab[:, :, 2]))
    sat_mean = float(np.mean(hsv[:, :, 1]))

    # 10. Micro-Text Stroke Irregularity
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    garbled = 0
    for cnt in contours:
        _, _, bw, bh = cv2.boundingRect(cnt)
        if 6 <= bh <= 15 and 4 <= bw <= 38:
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            if area > 8 and ((perim * perim) / (4.0 * np.pi * area)) > 6.8:
                garbled += 1

    return np.array([
        vae_jitter, mean_neutral_spread, edge_crispness, grid_ratio,
        ext_is_png, aspect_ratio, has_ai_meta, has_os_tag,
        ela_mean, ela_std, ela_p95,
        res_mean, res_std, flat_res_mean, flat_ela_mean,
        mid_freq, high_freq, fft_ratio,
        a_mean, a_std, b_mean, b_std, sat_mean,
        float(garbled), float(np.std(lap))
    ], dtype=np.float32)