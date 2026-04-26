import json
import math
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

warnings.filterwarnings("ignore")

# ==========================================================
# Fuji Look Assistant
# A realistic Fujifilm recipe direction + tuning assistant.
# ==========================================================

APP_VERSION = "1.3-black-gold-xtrans-all"

# Sensor-safe film simulation menus. Some individual camera bodies/firmware may vary,
# but these lists keep recommendations realistic for each X-Trans generation.
XTRANS_I_SIMS = [
    "Provia / Standard",
    "Velvia / Vivid",
    "Astia / Soft",
    "Pro Neg. Hi",
    "Pro Neg. Std",
    "Monochrome",
    "Monochrome + Ye Filter",
    "Monochrome + R Filter",
    "Monochrome + G Filter",
    "Sepia",
]

XTRANS_II_SIMS = [
    "Provia / Standard",
    "Velvia / Vivid",
    "Astia / Soft",
    "Classic Chrome",
    "Pro Neg. Hi",
    "Pro Neg. Std",
    "Monochrome",
    "Monochrome + Ye Filter",
    "Monochrome + R Filter",
    "Monochrome + G Filter",
    "Sepia",
]

XTRANS_III_SIMS = [
    "Provia / Standard",
    "Velvia / Vivid",
    "Astia / Soft",
    "Classic Chrome",
    "Pro Neg. Hi",
    "Pro Neg. Std",
    "Acros",
    "Acros + Ye Filter",
    "Acros + R Filter",
    "Acros + G Filter",
    "Monochrome",
    "Monochrome + Ye Filter",
    "Monochrome + R Filter",
    "Monochrome + G Filter",
    "Sepia",
]

XTRANS_IV_SIMS = [
    "Provia / Standard",
    "Velvia / Vivid",
    "Astia / Soft",
    "Classic Chrome",
    "Pro Neg. Hi",
    "Pro Neg. Std",
    "Classic Negative",
    "Eterna / Cinema",
    "Eterna Bleach Bypass",
    "Acros",
    "Acros + Ye Filter",
    "Acros + R Filter",
    "Acros + G Filter",
    "Monochrome",
    "Monochrome + Ye Filter",
    "Monochrome + R Filter",
    "Monochrome + G Filter",
    "Sepia",
]

XTRANS_V_ONLY = ["Nostalgic Negative", "Reala Ace"]
XTRANS_V_SIMS = [
    "Provia / Standard",
    "Velvia / Vivid",
    "Astia / Soft",
    "Classic Chrome",
    "Reala Ace",
    "Pro Neg. Hi",
    "Pro Neg. Std",
    "Classic Negative",
    "Nostalgic Negative",
    "Eterna / Cinema",
    "Eterna Bleach Bypass",
    "Acros",
    "Acros + Ye Filter",
    "Acros + R Filter",
    "Acros + G Filter",
    "Monochrome",
    "Monochrome + Ye Filter",
    "Monochrome + R Filter",
    "Monochrome + G Filter",
    "Sepia",
]

SENSOR_SIM_MAP = {
    "I": XTRANS_I_SIMS,
    "II": XTRANS_II_SIMS,
    "III": XTRANS_III_SIMS,
    "IV": XTRANS_IV_SIMS,
    "V": XTRANS_V_SIMS,
}

SENSOR_LABEL_MAP = {
    "I": "X-Trans I",
    "II": "X-Trans II",
    "III": "X-Trans III",
    "IV": "X-Trans IV",
    "V": "X-Trans V",
}

LEGACY_SENSORS = {"I", "II", "III"}


@dataclass
class RecipePreset:
    name: str
    family: str
    film_simulation: str
    sensor: str  # IV, V, BOTH
    grain_effect: str
    color_chrome_effect: str
    color_chrome_fx_blue: str
    white_balance: str
    wb_shift_r: int
    wb_shift_b: int
    dynamic_range: str
    highlights: int
    shadows: int
    color: int
    sharpness: int
    noise_reduction: int
    clarity: int
    iso: str
    exposure_comp: str
    target: Dict[str, float]
    best_for: str
    notes: str


def _preset(
    name: str,
    family: str,
    film: str,
    sensor: str,
    grain: str,
    cce: str,
    ccb: str,
    wb: str,
    r: int,
    b: int,
    dr: str,
    hi: int,
    sh: int,
    col: int,
    sharp: int,
    nr: int,
    clarity: int,
    iso: str,
    exp: str,
    target: Dict[str, float],
    best_for: str,
    notes: str,
) -> RecipePreset:
    return RecipePreset(
        name, family, film, sensor, grain, cce, ccb, wb, r, b, dr, hi, sh, col,
        sharp, nr, clarity, iso, exp, target, best_for, notes
    )


RECIPE_PRESETS: List[RecipePreset] = [
    _preset("Classic Negative - Moody Warm", "moody vintage", "Classic Negative", "BOTH", "Weak Small", "Weak", "Weak", "Auto", 2, -3, "DR400", -1, 2, 1, -1, -4, 1, "Auto up to 3200", "+1/3", {"warmth": .65, "tint": .50, "sat": .48, "contrast": .70, "brightness": .42, "softness": .40, "vintage": .85, "cinematic": .75}, "street, cafe, evening, lifestyle", "Classic Negative gives compressed shadows and a nostalgic colour separation."),
    _preset("Classic Negative - Cool Cinematic", "cool cinematic", "Classic Negative", "BOTH", "Strong Large", "Strong", "Strong", "Auto", -2, 3, "DR400", 1, 2, -1, -1, -4, 2, "Auto up to 6400", "-1/3", {"warmth": .28, "tint": .48, "sat": .36, "contrast": .82, "brightness": .35, "softness": .30, "vintage": .65, "cinematic": .95}, "night, neon, rain, moody interiors", "Cooler WB shift and FX Blue help preserve blue/cyan atmosphere."),
    _preset("Classic Chrome - Kodachrome Direction", "classic chrome vintage", "Classic Chrome", "BOTH", "Weak Small", "Strong", "Weak", "Daylight", 2, -4, "DR400", -2, 2, 2, -1, -4, 1, "Auto up to 1600", "+1/3", {"warmth": .58, "tint": .46, "sat": .52, "contrast": .66, "brightness": .50, "softness": .36, "vintage": .92, "cinematic": .65}, "travel, documentary, daylight scenes", "A safer Kodachrome-style starting point: warm, contrasty, slightly muted but not flat."),
    _preset("Classic Chrome - Documentary Soft", "documentary muted", "Classic Chrome", "BOTH", "Weak Small", "Weak", "Off", "Auto", 1, -2, "DR400", -2, 1, 0, -1, -3, 0, "Auto up to 1600", "+1/3", {"warmth": .54, "tint": .49, "sat": .38, "contrast": .48, "brightness": .55, "softness": .60, "vintage": .78, "cinematic": .55}, "documentary, everyday, travel", "Muted colour and soft highlights make it forgiving for mixed scenes."),
    _preset("Nostalgic Negative - Warm Print", "warm nostalgic print", "Nostalgic Negative", "V", "Weak Small", "Off", "Off", "Daylight", 3, -3, "DR400", -1, 2, 1, -2, -4, 2, "Auto up to 1600", "+1/3", {"warmth": .78, "tint": .52, "sat": .45, "contrast": .46, "brightness": .58, "softness": .62, "vintage": .96, "cinematic": .58}, "portraits, weddings, lifestyle, warm interiors", "X-Trans V warm print direction with amber highlights and gentle colour."),
    _preset("Reala Ace - Clean Realist", "clean realistic", "Reala Ace", "V", "Off", "Weak", "Off", "Auto", 0, -1, "DR200", -1, 0, 0, 0, -2, 0, "Auto up to 1600", "0", {"warmth": .52, "tint": .50, "sat": .50, "contrast": .50, "brightness": .53, "softness": .45, "vintage": .38, "cinematic": .35}, "product, portrait, cafe, clean editorial", "A natural base when the reference is not heavily stylised."),
    _preset("Eterna - Soft Cinema", "soft cinematic", "Eterna / Cinema", "BOTH", "Weak Small", "Off", "Weak", "Auto", -1, 2, "DR400", -2, 1, -2, -2, -3, -3, "Auto up to 3200", "+1/3", {"warmth": .42, "tint": .50, "sat": .28, "contrast": .34, "brightness": .48, "softness": .85, "vintage": .45, "cinematic": .90}, "video-like stills, portraits, fog, window light", "Use when the reference is flat, soft, muted, and cinematic."),
    _preset("Eterna Bleach Bypass - Hard Cinema", "hard cinematic", "Eterna Bleach Bypass", "BOTH", "Strong Large", "Strong", "Strong", "Auto", -1, 1, "DR400", 1, 3, -3, 1, -4, 3, "Auto up to 6400", "-1/3", {"warmth": .40, "tint": .48, "sat": .18, "contrast": .92, "brightness": .34, "softness": .18, "vintage": .50, "cinematic": .98}, "dramatic street, industrial, stormy scenes", "Low saturation, high contrast, gritty cinematic direction."),
    _preset("Pro Neg. Std - Soft Portrait", "soft portrait", "Pro Neg. Std", "BOTH", "Off", "Off", "Off", "Auto", 1, -1, "DR200", -2, 0, 0, -2, -2, -2, "Auto up to 1600", "+2/3", {"warmth": .55, "tint": .52, "sat": .36, "contrast": .28, "brightness": .62, "softness": .88, "vintage": .35, "cinematic": .40}, "skin, wedding, indoor portraits", "Soft contrast and gentle colour. Good when skin tone matters more than drama."),
    _preset("Astia - Bright Lifestyle", "bright soft", "Astia / Soft", "BOTH", "Off", "Weak", "Off", "Daylight", 1, -2, "DR200", -1, -1, 1, 0, -2, -1, "Auto up to 1600", "+1/3", {"warmth": .58, "tint": .51, "sat": .58, "contrast": .38, "brightness": .70, "softness": .72, "vintage": .28, "cinematic": .32}, "bright cafes, family, clean lifestyle", "Soft but slightly colourful; useful for bright airy references."),
    _preset("Velvia - Vivid Travel", "vivid punchy", "Velvia / Vivid", "BOTH", "Off", "Strong", "Weak", "Daylight", 0, 0, "DR100", -1, 0, 3, 1, -3, -1, "Auto up to 800", "0", {"warmth": .52, "tint": .50, "sat": .86, "contrast": .65, "brightness": .56, "softness": .25, "vintage": .20, "cinematic": .25}, "landscape, flowers, colourful food", "Use only when the reference is clearly saturated and punchy."),
    _preset("Provia - Neutral Base", "neutral balanced", "Provia / Standard", "BOTH", "Off", "Off", "Off", "Auto", 0, 0, "DR200", 0, 0, 0, 0, -2, 0, "Auto", "0", {"warmth": .50, "tint": .50, "sat": .50, "contrast": .50, "brightness": .52, "softness": .45, "vintage": .22, "cinematic": .25}, "general fallback, natural colour", "Choose this when the image is clean and not heavily stylised."),
    _preset("Acros - Deep Street B&W", "black white contrast", "Acros + R Filter", "BOTH", "Strong Large", "Off", "Off", "Auto", 0, 0, "DR400", 1, 3, 0, 2, -4, 3, "Auto up to 6400", "0", {"warmth": .50, "tint": .50, "sat": .02, "contrast": .85, "brightness": .40, "softness": .22, "vintage": .65, "cinematic": .80}, "B&W street, harsh light, portraits", "Strong monochrome direction. Use if the reference is already nearly colourless or very contrasty."),
    _preset("Acros - Soft B&W Portrait", "black white soft", "Acros + Ye Filter", "BOTH", "Weak Small", "Off", "Off", "Auto", 0, 0, "DR400", -1, 1, 0, 0, -3, 0, "Auto up to 3200", "+1/3", {"warmth": .50, "tint": .50, "sat": .03, "contrast": .45, "brightness": .58, "softness": .70, "vintage": .55, "cinematic": .55}, "soft B&W portraits, documentary", "Lower contrast B&W starting point."),
    _preset("Sepia - Aged Warm", "sepia vintage", "Sepia", "BOTH", "Weak Small", "Off", "Off", "Auto", 0, 0, "DR200", -1, 1, 0, -1, -2, 1, "Auto", "0", {"warmth": .82, "tint": .52, "sat": .16, "contrast": .46, "brightness": .52, "softness": .62, "vintage": .90, "cinematic": .42}, "intentionally aged warm monochrome", "Only use when the reference has a clear sepia/brown monochrome look."),

    # Legacy-friendly look directions for X-Trans I, II and III bodies.
    _preset("Pro Neg. Hi - Legacy Contrast", "legacy portrait contrast", "Pro Neg. Hi", "BOTH", "Off", "Off", "Off", "Auto", 1, -1, "DR200", 1, 1, 1, 0, -2, 0, "Auto up to 1600", "0", {"warmth": .55, "tint": .50, "sat": .48, "contrast": .64, "brightness": .50, "softness": .38, "vintage": .35, "cinematic": .38}, "older Fujifilm bodies, portraits, documentary, contrasty daylight", "A strong older-body option when Classic Negative or Eterna is unavailable."),
    _preset("Pro Neg. Std - Legacy Soft Print", "legacy soft print", "Pro Neg. Std", "BOTH", "Off", "Off", "Off", "Auto", 1, -2, "DR200", -1, 0, 0, -1, -2, 0, "Auto up to 1600", "+1/3", {"warmth": .58, "tint": .51, "sat": .35, "contrast": .34, "brightness": .60, "softness": .82, "vintage": .52, "cinematic": .42}, "X-Trans I/II/III portraits, weddings, gentle daylight", "A soft legacy starting point with gentle colour and forgiving contrast."),
    _preset("Monochrome - Legacy Deep Contrast", "legacy black white contrast", "Monochrome + R Filter", "BOTH", "Off", "Off", "Off", "Auto", 0, 0, "DR400", 1, 2, 0, 1, -2, 0, "Auto up to 3200", "0", {"warmth": .50, "tint": .50, "sat": .02, "contrast": .80, "brightness": .40, "softness": .26, "vintage": .58, "cinematic": .72}, "X-Trans I/II black-and-white street, harsh daylight, architecture", "Legacy monochrome alternative for bodies without Acros."),
    _preset("Monochrome - Legacy Soft B&W", "legacy black white soft", "Monochrome + Ye Filter", "BOTH", "Off", "Off", "Off", "Auto", 0, 0, "DR200", -1, 1, 0, 0, -2, 0, "Auto up to 1600", "+1/3", {"warmth": .50, "tint": .50, "sat": .03, "contrast": .42, "brightness": .58, "softness": .72, "vintage": .55, "cinematic": .50}, "X-Trans I/II soft B&W portraits and documentary", "A gentler monochrome option for older Fujifilm bodies."),
]

# Add small family variations programmatically without pretending they are exact film stocks.
VARIATIONS = [
    ("Soft", {"highlights": -1, "shadows": -1, "clarity": -1, "sharpness": -1}, {"contrast": -0.08, "softness": 0.10, "brightness": 0.03}),
    ("Crisp", {"highlights": 0, "shadows": 1, "clarity": 1, "sharpness": 1}, {"contrast": 0.08, "softness": -0.10, "brightness": -0.02}),
    ("Warm", {"wb_shift_r": 1, "wb_shift_b": -1}, {"warmth": 0.08}),
    ("Cool", {"wb_shift_r": -1, "wb_shift_b": 1}, {"warmth": -0.08}),
]


def expanded_presets() -> List[RecipePreset]:
    out = list(RECIPE_PRESETS)
    for base in RECIPE_PRESETS:
        if base.film_simulation in ["Sepia"] or "Acros" in base.film_simulation:
            continue
        for label, setting_delta, target_delta in VARIATIONS:
            p = RecipePreset(**asdict(base))
            p.name = f"{base.name} / {label} Tune"
            p.notes = f"Variation of {base.name}. {base.notes}"
            for k, delta in setting_delta.items():
                val = getattr(p, k) + delta
                if k in ["wb_shift_r", "wb_shift_b"]:
                    val = int(np.clip(val, -9, 9))
                elif k in ["highlights", "shadows", "color", "sharpness", "noise_reduction", "clarity"]:
                    val = int(np.clip(val, -4, 4))
                setattr(p, k, val)
            for k, delta in target_delta.items():
                p.target[k] = float(np.clip(p.target.get(k, 0.5) + delta, 0.0, 1.0))
            out.append(p)
    return out


ALL_PRESETS = expanded_presets()


# ---------------- Image Analysis ----------------
def load_rgb(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Resize for stable speed, not visual display.
    max_side = 1100
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return np.asarray(image).astype(np.uint8)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    if HAS_CV2:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # Fallback approximation with PIL/NumPy if OpenCV is unavailable.
    arr = rgb.astype(np.float32) / 255.0
    L = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    a = arr[:, :, 0] - arr[:, :, 1]
    b = arr[:, :, 1] - arr[:, :, 2]
    return np.stack([L * 255.0, (a + 1) * 127.5, (b + 1) * 127.5], axis=2).astype(np.float32)


def extract_look_features(rgb: np.ndarray) -> Dict[str, Any]:
    lab = rgb_to_lab(rgb)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0

    arr = rgb.astype(np.float32) / 255.0
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    l_mean = float(np.mean(L) / 255.0)
    l_p5, l_p25, l_p50, l_p75, l_p95 = [float(np.percentile(L, p) / 255.0) for p in [5, 25, 50, 75, 95]]
    contrast = float(np.clip((l_p95 - l_p5) * 1.45, 0, 1))
    softness = float(np.clip(1.0 - contrast, 0, 1))

    # Saturation/chroma from RGB and Lab.
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    sat = np.where(maxc <= 1e-6, 0, (maxc - minc) / np.maximum(maxc, 1e-6))
    saturation = float(np.clip(np.mean(sat) * 1.18, 0, 1))
    chroma = np.sqrt(a * a + b * b)
    colorfulness = float(np.clip(np.mean(chroma) / 55.0, 0, 1))

    warmth_raw = float((np.mean(R) - np.mean(B)) * 1.3)
    warmth = float(np.clip(0.5 + warmth_raw, 0, 1))
    tint_raw = float((np.mean(R) + np.mean(B)) / 2.0 - np.mean(G))
    tint = float(np.clip(0.5 + tint_raw * 1.5, 0, 1))  # above .5 magenta, below .5 green

    # Black & white / low colour detection.
    low_colour = float(np.clip(1.0 - (saturation * 1.8 + colorfulness * 0.8), 0, 1))

    # Highlight/shadow colour casts.
    shadow_mask = L <= np.percentile(L, 25)
    high_mask = L >= np.percentile(L, 75)
    def masked_mean(channel, mask, default=0.0):
        return float(np.mean(channel[mask])) if np.any(mask) else default

    shadow_warmth = float(np.clip(0.5 + (masked_mean(R, shadow_mask) - masked_mean(B, shadow_mask)) * 1.4, 0, 1))
    highlight_warmth = float(np.clip(0.5 + (masked_mean(R, high_mask) - masked_mean(B, high_mask)) * 1.4, 0, 1))

    # Texture/sharpness safely.
    if HAS_CV2:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = cv2.Laplacian(np.ascontiguousarray(gray), cv2.CV_32F)
        sharpness = float(np.clip(np.var(lap) / 900.0, 0, 1))
        edges = cv2.Canny(rgb, 80, 160)
        edge_density = float(np.mean(edges > 0))
    else:
        gray = np.mean(arr, axis=2)
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        grad = np.sqrt(gx * gx + gy * gy)
        sharpness = float(np.clip(np.var(grad) * 25, 0, 1))
        edge_density = float(np.clip(np.mean(grad > np.percentile(grad, 85)), 0, 1))

    softness = float(np.clip((softness * .70) + ((1 - sharpness) * .30), 0, 1))

    # Cinematic and vintage are style estimates, not hard truth.
    cinematic = float(np.clip((contrast * .35) + ((1 - saturation) * .35) + ((1 - l_mean) * .20) + (edge_density * .10), 0, 1))
    vintage = float(np.clip((low_colour * .25) + (abs(warmth - .5) * .55) + ((1 - saturation) * .20) + (contrast * .15), 0, 1))

    # Scene tags.
    tags = []
    if saturation < .12 or colorfulness < .12:
        tags.append("near monochrome")
    if warmth > .63:
        tags.append("warm")
    if warmth < .38:
        tags.append("cool")
    if contrast > .68:
        tags.append("contrasty")
    if contrast < .38:
        tags.append("low contrast")
    if saturation > .62:
        tags.append("vibrant")
    if saturation < .32:
        tags.append("muted")
    if l_mean > .65:
        tags.append("bright")
    if l_mean < .38:
        tags.append("dark")
    if softness > .68:
        tags.append("soft")
    if cinematic > .70:
        tags.append("cinematic")
    if vintage > .70:
        tags.append("vintage")

    palette = dominant_palette(rgb, k=5)

    return {
        "warmth": warmth,
        "tint": tint,
        "sat": saturation,
        "colorfulness": colorfulness,
        "contrast": contrast,
        "brightness": l_mean,
        "softness": softness,
        "sharpness": sharpness,
        "cinematic": cinematic,
        "vintage": vintage,
        "low_colour": low_colour,
        "shadow_warmth": shadow_warmth,
        "highlight_warmth": highlight_warmth,
        "luminance_p5": l_p5,
        "luminance_p50": l_p50,
        "luminance_p95": l_p95,
        "tags": tags,
        "palette": palette,
    }


def dominant_palette(rgb: np.ndarray, k: int = 5) -> List[str]:
    # lightweight palette: quantize RGB into bins and return hex centers.
    small = rgb.reshape(-1, 3)
    if small.shape[0] > 60000:
        idx = np.linspace(0, small.shape[0] - 1, 60000).astype(int)
        small = small[idx]
    bins = (small // 32).astype(np.int32)
    packed = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    values, counts = np.unique(packed, return_counts=True)
    top = values[np.argsort(counts)[-k:]][::-1]
    hexes = []
    for v in top:
        rbin = v // 64
        gbin = (v % 64) // 8
        bbin = v % 8
        center = np.array([rbin, gbin, bbin]) * 32 + 16
        center = np.clip(center, 0, 255).astype(int)
        hexes.append("#%02x%02x%02x" % tuple(center))
    return hexes


# ---------------- Matching + Tuning ----------------
def sensor_presets(sensor_code: str) -> List[RecipePreset]:
    sims = SENSOR_SIM_MAP.get(sensor_code, XTRANS_V_SIMS)
    return [p for p in ALL_PRESETS if (p.sensor in ["BOTH", "ALL", sensor_code]) and p.film_simulation in sims]


def format_signed(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{int(value):+d}"
    return str(value)


def sensor_capability_note(sensor_code: str) -> str:
    if sensor_code == "I":
        return "X-Trans I mode removes newer JPEG settings such as Acros, Classic Chrome, Color Chrome, Grain size and Clarity. Tone/colour values are kept within older-body ranges."
    if sensor_code == "II":
        return "X-Trans II mode keeps older-body settings conservative. Classic Chrome availability can depend on camera/firmware, while Color Chrome FX Blue and Clarity are removed."
    if sensor_code == "III":
        return "X-Trans III mode removes Color Chrome FX Blue and Clarity. Grain size is simplified to Off / Weak / Strong."
    if sensor_code == "IV":
        return "X-Trans IV mode keeps modern controls such as Grain size, Color Chrome Effect, Color Chrome FX Blue and Clarity, while excluding X-Trans V-only simulations."
    return "X-Trans V mode enables the widest simulation set, including Nostalgic Negative and Reala Ace where applicable."


def simplify_grain_for_sensor(grain: str, sensor_code: str) -> str:
    if sensor_code in ["I", "II"]:
        return "Not available"
    if sensor_code == "III":
        if "Strong" in grain:
            return "Strong"
        if "Weak" in grain:
            return "Weak"
        return "Off"
    return grain


def sanitize_recipe_for_sensor(recipe: Dict[str, Any], sensor_code: str) -> Dict[str, Any]:
    r = dict(recipe)
    r["sensor_generation"] = SENSOR_LABEL_MAP.get(sensor_code, sensor_code)
    r["sensor_note"] = sensor_capability_note(sensor_code)

    if sensor_code in LEGACY_SENSORS:
        for key in ["highlights", "shadows", "color", "sharpness", "noise_reduction"]:
            if isinstance(r.get(key), (int, np.integer)):
                r[key] = int(np.clip(r[key], -2, 2))
        r["color_chrome_effect"] = "Not available"
        r["color_chrome_fx_blue"] = "Not available"
        r["clarity"] = "Not available"
        r["grain_effect"] = simplify_grain_for_sensor(r.get("grain_effect", "Off"), sensor_code)

    if r.get("film_simulation") not in SENSOR_SIM_MAP.get(sensor_code, []):
        r["film_simulation"] = "Provia / Standard"
        r["name"] = f"{r.get('name', 'Custom')} / Legacy Fallback"

    return r


def score_preset(features: Dict[str, Any], preset: RecipePreset) -> Tuple[float, Dict[str, float]]:
    weights = {
        "warmth": 1.15,
        "tint": 0.45,
        "sat": 1.25,
        "contrast": 1.15,
        "brightness": 0.70,
        "softness": 0.80,
        "vintage": 0.65,
        "cinematic": 0.65,
    }
    details = {}
    score = 0.0
    total = 0.0
    for k, w in weights.items():
        fv = float(features.get(k, 0.5))
        tv = float(preset.target.get(k, 0.5))
        # Wider sigma: this is a direction finder, not a fake exact detector.
        sigma = 0.22 if k in ["sat", "contrast", "warmth"] else 0.28
        sim = math.exp(-((fv - tv) ** 2) / (2 * sigma * sigma))
        details[k] = sim
        score += sim * w
        total += w

    # Strong rule for monochrome.
    is_bw_preset = "Acros" in preset.film_simulation or "Monochrome" in preset.film_simulation or preset.film_simulation == "Sepia"
    if features.get("low_colour", 0) > .72 and is_bw_preset:
        score += .25 * total
    if features.get("low_colour", 0) < .45 and is_bw_preset:
        score -= .35 * total
    if features.get("sat", 0.5) > .68 and preset.film_simulation in ["Velvia / Vivid"]:
        score += .15 * total

    final = float(np.clip(score / total, 0, 1))
    return final, details


def recommend(features: Dict[str, Any], sensor_code: str) -> Dict[str, Any]:
    candidates = []
    for p in sensor_presets(sensor_code):
        s, d = score_preset(features, p)
        candidates.append({"preset": p, "score": s, "details": d})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    alts = candidates[1:5]

    recipe = tune_recipe_from_features(best["preset"], features)
    recipe = sanitize_recipe_for_sensor(recipe, sensor_code)
    guidance = build_guidance(features, recipe)
    summary = visual_summary(features)

    return {
        "best": recipe,
        "best_score": best["score"],
        "alternatives": [{"name": c["preset"].name, "film_simulation": c["preset"].film_simulation, "fit": c["score"], "best_for": c["preset"].best_for} for c in alts],
        "visual_summary": summary,
        "guidance": guidance,
        "features": features,
    }


def tune_recipe_from_features(p: RecipePreset, features: Dict[str, Any]) -> Dict[str, Any]:
    recipe = asdict(p)
    recipe.pop("target", None)

    # Generate a more responsive starting point from image features.
    warmth = features.get("warmth", .5)
    tint = features.get("tint", .5)
    sat = features.get("sat", .5)
    contrast = features.get("contrast", .5)
    brightness = features.get("brightness", .5)
    softness = features.get("softness", .5)
    low_colour = features.get("low_colour", .0)

    if warmth > .68:
        recipe["wb_shift_r"] = int(np.clip(recipe["wb_shift_r"] + 1, -9, 9))
        recipe["wb_shift_b"] = int(np.clip(recipe["wb_shift_b"] - 1, -9, 9))
    elif warmth < .36:
        recipe["wb_shift_r"] = int(np.clip(recipe["wb_shift_r"] - 1, -9, 9))
        recipe["wb_shift_b"] = int(np.clip(recipe["wb_shift_b"] + 1, -9, 9))

    if tint < .43:  # green leaning reference
        recipe["wb_shift_r"] = int(np.clip(recipe["wb_shift_r"] - 1, -9, 9))
    elif tint > .58:  # magenta leaning reference
        recipe["wb_shift_r"] = int(np.clip(recipe["wb_shift_r"] + 1, -9, 9))

    if sat > .64 and low_colour < .3:
        recipe["color"] = int(np.clip(recipe["color"] + 1, -4, 4))
    elif sat < .30 or low_colour > .65:
        recipe["color"] = int(np.clip(recipe["color"] - 1, -4, 4))

    if contrast > .70:
        recipe["shadows"] = int(np.clip(recipe["shadows"] + 1, -4, 4))
        recipe["highlights"] = int(np.clip(recipe["highlights"] + 0, -4, 4))
    elif contrast < .38:
        recipe["shadows"] = int(np.clip(recipe["shadows"] - 1, -4, 4))
        recipe["highlights"] = int(np.clip(recipe["highlights"] - 1, -4, 4))

    if brightness > .67:
        recipe["highlights"] = int(np.clip(recipe["highlights"] - 1, -4, 4))
        recipe["exposure_comp"] = "+1/3"
    elif brightness < .35:
        recipe["shadows"] = int(np.clip(recipe["shadows"] + 1, -4, 4))
        recipe["exposure_comp"] = "-1/3"

    if softness > .70:
        recipe["clarity"] = int(np.clip(recipe["clarity"] - 1, -5, 5))
        recipe["sharpness"] = int(np.clip(recipe["sharpness"] - 1, -4, 4))
    elif softness < .32:
        recipe["clarity"] = int(np.clip(recipe["clarity"] + 1, -5, 5))
        recipe["sharpness"] = int(np.clip(recipe["sharpness"] + 1, -4, 4))

    # DR selection by highlight/shadow spread.
    drange = features.get("luminance_p95", .8) - features.get("luminance_p5", .1)
    if drange > .72:
        recipe["dynamic_range"] = "DR400"
    elif drange > .55:
        recipe["dynamic_range"] = "DR200"
    else:
        recipe["dynamic_range"] = "DR100"

    return recipe


def visual_summary(features: Dict[str, Any]) -> str:
    tags = features.get("tags", [])
    if not tags:
        return "neutral / balanced / natural"
    # keep ordered, unique
    seen = []
    for t in tags:
        if t not in seen:
            seen.append(t)
    return " / ".join(seen[:7])


def fit_label(score: float) -> str:
    if score >= .82:
        return "Strong starting point"
    if score >= .68:
        return "Good starting point"
    if score >= .52:
        return "Usable direction"
    return "Experimental direction"


def build_guidance(features: Dict[str, Any], recipe: Dict[str, Any]) -> List[str]:
    lines = []
    if features.get("warmth", .5) > .64:
        lines.append("Reference is warm: keep WB shifted toward red/yellow. If your result becomes too orange, reduce R by 1 or add B +1.")
    elif features.get("warmth", .5) < .38:
        lines.append("Reference is cool: keep WB shifted toward blue. If skin looks too cold, add R +1.")
    else:
        lines.append("Reference colour temperature is fairly neutral: make small WB changes only.")

    if features.get("tint", .5) < .44:
        lines.append("There is a slight green bias: if your JPEG is too magenta, reduce WB R by 1; if it is too green, increase R by 1.")
    elif features.get("tint", .5) > .58:
        lines.append("There is a slight magenta bias: if the JPEG feels too pink, reduce WB R by 1.")

    if features.get("contrast", .5) > .68:
        lines.append("Reference is contrasty: raise Shadow if the image looks flat; lower Highlight if bright areas clip too quickly.")
    elif features.get("contrast", .5) < .40:
        lines.append("Reference is soft/low contrast: lower Shadow and Highlight for a gentler tonal curve.")

    if features.get("sat", .5) > .62:
        lines.append("Reference is colourful: increase Color only if your result is dull; avoid Velvia unless the look is intentionally punchy.")
    elif features.get("sat", .5) < .32:
        lines.append("Reference is muted: reduce Color by 1 if your result looks too digital or too saturated.")

    if features.get("softness", .5) > .68:
        lines.append("Reference is soft: use lower Clarity/Sharpness. Avoid over-sharpening skin or haze.")
    elif features.get("softness", .5) < .35:
        lines.append("Reference is crisp: use a little more Clarity/Sharpness, but keep NR low for texture.")

    return lines



def recipe_txt(recipe: Dict[str, Any], result: Dict[str, Any] = None) -> str:
    lines = []
    lines.append("FUJI LOOK ASSISTANT")
    lines.append("Generated Fujifilm Starting Recipe")
    lines.append("=" * 42)
    lines.append(f"Look Direction : {recipe.get('name')}")
    lines.append(f"Film Simulation: {recipe.get('film_simulation')}")
    lines.append(f"Sensor         : {recipe.get('sensor', 'BOTH')}")
    lines.append("")
    lines.append("SETTINGS")
    lines.append("-" * 42)
    lines.append(f"Dynamic Range        : {recipe.get('dynamic_range')}")
    lines.append(f"White Balance        : {recipe.get('white_balance')}")
    lines.append(f"WB Shift             : R {recipe.get('wb_shift_r'):+d}, B {recipe.get('wb_shift_b'):+d}")
    lines.append(f"Highlight            : {format_signed(recipe.get('highlights'))}")
    lines.append(f"Shadow               : {format_signed(recipe.get('shadows'))}")
    lines.append(f"Color                : {format_signed(recipe.get('color'))}")
    lines.append(f"Sharpness            : {format_signed(recipe.get('sharpness'))}")
    lines.append(f"Noise Reduction      : {format_signed(recipe.get('noise_reduction'))}")
    lines.append(f"Clarity              : {format_signed(recipe.get('clarity'))}")
    lines.append(f"Grain Effect         : {recipe.get('grain_effect')}")
    lines.append(f"Color Chrome Effect  : {recipe.get('color_chrome_effect')}")
    lines.append(f"Color Chrome FX Blue : {recipe.get('color_chrome_fx_blue')}")
    lines.append(f"ISO                  : {recipe.get('iso')}")
    lines.append(f"Exposure Comp        : {recipe.get('exposure_comp')}")
    lines.append("")
    lines.append("BEST FOR")
    lines.append("-" * 42)
    lines.append(recipe.get("best_for", ""))
    lines.append("")
    lines.append("NOTES")
    lines.append("-" * 42)
    lines.append(recipe.get("notes", ""))
    if result:
        lines.append("")
        lines.append("VISUAL DIAGNOSIS")
        lines.append("-" * 42)
        lines.append(result.get("visual_summary", ""))
        lines.append("")
        lines.append("LOOK NOTES")
        lines.append("-" * 42)
        for g in result.get("guidance", []):
            lines.append(f"- {g}")
    lines.append("")
    lines.append("Important: This is a starting recipe direction, not a guaranteed exact reverse-engineered match.")
    return "\n".join(lines)


# ---------------- UI ----------------
st.set_page_config(
    page_title="Fuji Look Assistant",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700;800;900&family=Inter:wght@400;500;600;700;800;900&display=swap');
:root { --black:#020202; --graphite:#101010; --gold:#d6aa45; --gold-2:#f2d37a; --gold-dark:#8f6721; --text:#f7f1e3; --muted:#b8ad98; --dim:#7f7465; --line:rgba(214,170,69,.20); --line-strong:rgba(242,211,122,.38); --panel:rgba(11,11,11,.96); }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp { background: radial-gradient(circle at 18% 0%, rgba(214,170,69,.16), transparent 28%), radial-gradient(circle at 82% 12%, rgba(242,211,122,.08), transparent 26%), linear-gradient(135deg,#000 0%,#050505 44%,#101010 100%); color:var(--text); }
.block-container { padding-top:2rem; max-width:1340px; }
.hero { position:relative; overflow:hidden; background:linear-gradient(135deg,rgba(6,6,6,.98) 0%,rgba(13,13,13,.98) 54%,rgba(28,20,8,.96) 100%); border-radius:30px; padding:42px 44px; color:var(--text); border:1px solid var(--line-strong); box-shadow:0 34px 110px rgba(0,0,0,.62), inset 0 1px 0 rgba(255,255,255,.04); margin-bottom:24px; }
.hero:before { content:""; position:absolute; inset:0; background:linear-gradient(90deg,transparent 0%,rgba(242,211,122,.08) 50%,transparent 100%), repeating-linear-gradient(90deg,rgba(255,255,255,.018) 0,rgba(255,255,255,.018) 1px,transparent 1px,transparent 16px); pointer-events:none; }
.hero:after { content:""; position:absolute; right:-90px; bottom:-110px; width:360px; height:360px; border:1px solid rgba(214,170,69,.18); border-radius:50%; box-shadow:0 0 0 28px rgba(214,170,69,.025), 0 0 0 62px rgba(214,170,69,.018); pointer-events:none; }
.hero-inner { position:relative; z-index:1; }
.hero h1 { font-family:'Cinzel',serif; font-size:3.15rem; line-height:1.02; margin:0 0 14px 0; letter-spacing:-.035em; font-weight:900; background:linear-gradient(180deg,#fff8e5 0%,#d6aa45 48%,#8f6721 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { color:rgba(247,241,227,.76); font-size:1.06rem; margin:0; max-width:890px; }
.lux-mark { display:inline-flex; align-items:center; gap:10px; background:linear-gradient(135deg,rgba(214,170,69,.18),rgba(242,211,122,.06)); border:1px solid var(--line-strong); color:var(--gold-2); border-radius:999px; padding:8px 14px; font-weight:900; font-size:.78rem; margin-bottom:20px; letter-spacing:.16em; text-transform:uppercase; box-shadow:0 12px 28px rgba(0,0,0,.28); }
.card { background:linear-gradient(180deg,var(--panel) 0%,rgba(5,5,5,.97) 100%); border:1px solid var(--line); border-radius:26px; padding:25px; box-shadow:0 22px 64px rgba(0,0,0,.48), inset 0 1px 0 rgba(255,255,255,.035); color:var(--text); }
.metric-card { background:linear-gradient(180deg,rgba(214,170,69,.08) 0%,rgba(255,255,255,.025) 100%); border:1px solid rgba(214,170,69,.18); border-radius:18px; padding:15px 16px; min-height:90px; }
.metric-label { color:var(--dim); font-size:.74rem; text-transform:uppercase; letter-spacing:.11em; font-weight:900; }
.metric-value { color:var(--text); font-size:1.18rem; font-weight:900; margin-top:6px; }
.recipe-title { font-family:'Cinzel',serif; color:var(--gold-2); font-size:1.92rem; letter-spacing:-.025em; line-height:1.08; margin:12px 0 6px 0; font-weight:800; }
.badge { display:inline-block; padding:7px 11px; border-radius:999px; background:rgba(214,170,69,.10); color:var(--gold-2); font-size:.75rem; font-weight:900; border:1px solid rgba(214,170,69,.28); letter-spacing:.04em; }
.fit-badge { background:linear-gradient(135deg,var(--gold) 0%,var(--gold-dark) 100%); color:#080808; border-color:rgba(242,211,122,.52); }
.setting-row { display:flex; justify-content:space-between; align-items:center; gap:16px; border-bottom:1px solid rgba(214,170,69,.13); padding:12px 0; }
.setting-label { color:var(--muted); font-weight:650; font-size:.92rem; }
.setting-value { color:var(--text); font-weight:900; font-size:.98rem; text-align:right; }
.palette-dot { display:inline-block; width:34px; height:34px; border-radius:50%; border:2px solid rgba(214,170,69,.36); margin-right:7px; vertical-align:middle; box-shadow:0 9px 20px rgba(0,0,0,.42); }
.small-muted { color:var(--muted); font-size:.93rem; }
.stButton>button, .stDownloadButton>button { border-radius:999px!important; font-weight:900!important; border:1px solid rgba(214,170,69,.30)!important; background:linear-gradient(180deg,rgba(214,170,69,.10),rgba(255,255,255,.025))!important; color:var(--text)!important; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:rgba(242,211,122,.75)!important; color:var(--gold-2)!important; transform:translateY(-1px); }
.stButton>button[kind="primary"] { background:linear-gradient(135deg,var(--gold-2) 0%,var(--gold) 42%,var(--gold-dark) 100%)!important; color:#070707!important; border-color:rgba(255,255,255,.14)!important; box-shadow:0 16px 38px rgba(214,170,69,.18)!important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#000 0%,#080808 54%,#111 100%); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] * { color:var(--text); }
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] p { color:var(--muted)!important; }
section[data-testid="stSidebar"] .stRadio label { color:var(--text)!important; }
hr { border:none; border-top:1px solid var(--line); }
h1,h2,h3,[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3 { color:var(--text); }
[data-testid="stFileUploader"] { background:rgba(214,170,69,.045); border-radius:20px; border:1px solid rgba(214,170,69,.18); padding:8px; }
[data-testid="stAlert"] { background:rgba(214,170,69,.09); border:1px solid rgba(214,170,69,.20); color:var(--text); }
a { color:var(--gold-2)!important; text-decoration:none; }
a:hover { color:#fff2c2!important; text-decoration:none; }
.social-footer { margin-top:22px; padding:28px 30px; border-radius:28px; background:linear-gradient(135deg,rgba(6,6,6,.98),rgba(20,16,8,.96)); border:1px solid var(--line-strong); box-shadow:0 24px 70px rgba(0,0,0,.48), inset 0 1px 0 rgba(255,255,255,.035); }
.social-footer h3 { font-family:'Cinzel',serif; color:var(--gold-2); margin:0 0 8px 0; font-size:1.35rem; }
.social-footer p { color:var(--muted); margin:0 0 18px 0; }
.social-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }
.social-card { display:flex; align-items:center; gap:12px; padding:14px 15px; border-radius:18px; background:rgba(255,255,255,.035); border:1px solid rgba(214,170,69,.18); color:var(--text)!important; transition:all .18s ease; }
.social-card:hover { transform:translateY(-2px); border-color:rgba(242,211,122,.62); background:rgba(214,170,69,.08); }
.social-icon { width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,var(--gold-2),var(--gold-dark)); color:#060606; font-weight:950; flex:0 0 auto; }
.social-name { display:block; font-weight:900; font-size:.94rem; }
.social-handle { display:block; color:var(--muted); font-size:.78rem; margin-top:2px; }
.footer-note { color:var(--dim); font-size:.78rem; margin-top:16px; }
@media (max-width: 900px) { .hero h1{font-size:2.2rem;} .hero{padding:30px 26px;} .social-grid{grid-template-columns:1fr 1fr;} }
@media (max-width: 560px) { .social-grid{grid-template-columns:1fr;} }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="hero-inner">
        <div class="lux-mark">◆ FUJI LOOK ASSISTANT</div>
        <h1>Premium Fujifilm JPEG<br/>look direction studio.</h1>
        <p>Upload a reference image and receive a refined Fujifilm recipe direction, visual diagnosis, and sensor-safe starting point for X-Trans I, II, III, IV, or V.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🎛️ Camera Target")
    sensor_choice = st.radio("Sensor generation", ["X-Trans V", "X-Trans IV", "X-Trans III", "X-Trans II", "X-Trans I"], index=0)
    sensor_code = sensor_choice.replace("X-Trans ", "")
    sims = SENSOR_SIM_MAP.get(sensor_code, XTRANS_V_SIMS)
    st.markdown(f"**{len(sims)}** film simulation menu items")
    st.markdown(f"**{len(sensor_presets(sensor_code))}** look directions")
    st.caption(sensor_capability_note(sensor_code))
    st.divider()
    st.markdown("## 🧭 Studio Note")
    st.caption("Fuji Look Assistant recommends a refined starting point based on colour, tone, contrast, mood, and sensor compatibility. Older X-Trans bodies automatically hide unsupported newer settings.")

left, right = st.columns([.92, 1.08], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Upload reference image")
    uploaded = st.file_uploader("JPG, PNG, or WEBP", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        display_image = Image.open(uploaded)
        display_image = ImageOps.exif_transpose(display_image).convert("RGB")
        st.image(display_image, caption="Reference image", width="stretch")
        # Reset file pointer for analysis.
        uploaded.seek(0)
        if st.button("Analyze Look", type="primary", width="stretch"):
            rgb = load_rgb(uploaded)
            features = extract_look_features(rgb)
            result = recommend(features, sensor_code)
            st.session_state["result"] = result
            st.session_state["sensor_code"] = sensor_code
    else:
        st.info("Upload a reference photo to generate a Fujifilm look direction.")
    st.markdown('</div>', unsafe_allow_html=True)

    if "result" in st.session_state:
        result = st.session_state["result"]
        feats = result["features"]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("2. Visual diagnosis")
        st.markdown(f"<span class='badge'>{result['visual_summary']}</span>", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        metrics = [
            ("Warmth", feats["warmth"]), ("Saturation", feats["sat"]),
            ("Contrast", feats["contrast"]), ("Softness", feats["softness"]),
            ("Vintage", feats["vintage"]), ("Cinematic", feats["cinematic"]),
        ]
        for i, (label, val) in enumerate(metrics):
            with (c1 if i % 2 == 0 else c2):
                st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{val:.0%}</div></div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        palette_html = "".join([f"<span class='palette-dot' style='background:{h}' title='{h}'></span>" for h in feats.get("palette", [])])
        st.markdown("**Dominant palette**")
        st.markdown(palette_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    if "result" not in st.session_state:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What you’ll get")
        st.markdown("""
        - Closest Fujifilm **look direction**
        - Sensor-safe recipe for **X-Trans I, II, III, IV, or V**
        - Top alternative film simulations
        - Practical notes for the selected look direction
        - TXT + JSON export
        """)
        st.warning("This app intentionally avoids fake exact-match confidence. It gives a strong starting recipe direction based on visual analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        result = st.session_state["result"]
        recipe = result["best"]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<span class='badge fit-badge'>{fit_label(result['best_score'])}</span>", unsafe_allow_html=True)
        st.markdown(f"<h2 class='recipe-title'>{recipe['name']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='small-muted'>{recipe.get('notes','')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='small-muted'><strong>Sensor compatibility:</strong> {recipe.get('sensor_note', '')}</p>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Film Sim</div><div class='metric-value'>{recipe['film_simulation']}</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Fit</div><div class='metric-value'>{result['best_score']:.0%}</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Sensor</div><div class='metric-value'>{sensor_choice}</div></div>", unsafe_allow_html=True)

        st.markdown("### Recipe settings")
        sections = {
            "Colour": [
                ("Film Simulation", recipe["film_simulation"]),
                ("Color Chrome Effect", recipe["color_chrome_effect"]),
                ("Color Chrome FX Blue", recipe["color_chrome_fx_blue"]),
                ("Color", format_signed(recipe["color"])),
            ],
            "Tone": [
                ("Dynamic Range", recipe["dynamic_range"]),
                ("Highlight", format_signed(recipe["highlights"])),
                ("Shadow", format_signed(recipe["shadows"])),
                ("Exposure Comp", recipe["exposure_comp"]),
            ],
            "White Balance": [
                ("WB Type", recipe["white_balance"]),
                ("WB Shift", f"R {recipe['wb_shift_r']:+d}, B {recipe['wb_shift_b']:+d}"),
                ("ISO", recipe["iso"]),
            ],
            "Texture": [
                ("Grain Effect", recipe["grain_effect"]),
                ("Sharpness", format_signed(recipe["sharpness"])),
                ("Noise Reduction", format_signed(recipe["noise_reduction"])),
                ("Clarity", format_signed(recipe["clarity"])),
            ],
        }
        for title, rows in sections.items():
            st.markdown(f"**{title}**")
            for label, value in rows:
                st.markdown(f"<div class='setting-row'><span class='setting-label'>{label}</span><span class='setting-value'>{value}</span></div>", unsafe_allow_html=True)

        st.markdown("### Look notes")
        for g in result.get("guidance", []):
            st.markdown(f"- {g}")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top alternatives")
        for alt in result.get("alternatives", [])[:4]:
            st.markdown(f"<div class='setting-row'><span class='setting-label'>{alt['name']}<br><small>{alt['best_for']}</small></span><span class='setting-value'>{alt['film_simulation']}<br>{alt['fit']:.0%}</span></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Export")
        export_json = {
            "app": "Fuji Look Assistant",
            "version": APP_VERSION,
            "sensor": sensor_choice,
            "recipe": recipe,
            "visual_summary": result["visual_summary"],
            "guidance": result["guidance"],
            "alternatives": result["alternatives"],
            "note": "Starting recipe direction, not guaranteed exact reverse-engineered match.",
        }
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download TXT", recipe_txt(recipe, result), file_name="fuji_look_recipe.txt", mime="text/plain", width="stretch")
        with d2:
            st.download_button("Download JSON", json.dumps(export_json, indent=2), file_name="fuji_look_recipe.json", mime="application/json", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="social-footer">
      <h3>Connect with Halim</h3>
      <p>Follow the project, share feedback, or reach out for Fujifilm JPEG look experiments and visual direction.</p>
      <div class="social-grid">
        <a class="social-card" href="https://www.facebook.com/halim91/" target="_blank">
          <span class="social-icon">f</span>
          <span><span class="social-name">Facebook</span><span class="social-handle">halim91</span></span>
        </a>
        <a class="social-card" href="https://www.instagram.com/halimmok" target="_blank">
          <span class="social-icon">◎</span>
          <span><span class="social-name">Instagram</span><span class="social-handle">@halimmok</span></span>
        </a>
        <a class="social-card" href="https://www.threads.com/@halimmok" target="_blank">
          <span class="social-icon">@</span>
          <span><span class="social-name">Threads</span><span class="social-handle">@halimmok</span></span>
        </a>
        <a class="social-card" href="mailto:halim.jamal91@gmail.com">
          <span class="social-icon">✉</span>
          <span><span class="social-name">Gmail</span><span class="social-handle">halim.jamal91@gmail.com</span></span>
        </a>
      </div>
      <div class="footer-note">Fuji Look Assistant is a practical recipe direction tool. Real camera output still depends on lighting, exposure, lens, sensor generation, white balance, camera body/firmware, and the JPEG engine.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
