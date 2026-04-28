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

try:
    import colour
    HAS_COLOUR = True
except Exception:
    colour = None
    HAS_COLOUR = False

warnings.filterwarnings("ignore")

# ==========================================================
# Fuji Look Assistant
# A realistic Fujifilm recipe direction + tuning assistant.
# ==========================================================

APP_VERSION = "2.4-tone-color-safe"

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
    "IV_EARLY": XTRANS_IV_SIMS,
    "V": XTRANS_V_SIMS,
}

SENSOR_LABEL_MAP = {
    "I": "X-Trans I",
    "II": "X-Trans II",
    "III": "X-Trans III",
    "IV": "X-Trans IV",
    "IV_EARLY": "X-Trans IV Early",
    "V": "X-Trans V",
}

# Camera model selector. This is intentionally conservative because body firmware can vary.
CAMERA_MODEL_MAP = {
    "Auto / choose by sensor generation": None,
    "X-Pro1": "I", "X-E1": "I", "X-M1": "I",
    "X100S": "II", "X100T": "II", "X-T1": "II", "X-T10": "II", "X-E2 / X-E2S": "II", "X70": "II",
    "X-Pro2": "III", "X-T2": "III", "X-T20": "III", "X-E3": "III", "X100F": "III", "X-H1": "III",
    "X-T3 / X-T30": "IV_EARLY", "X-Pro3": "IV", "X100V": "IV", "X-T4": "IV", "X-S10": "IV", "X-E4": "IV", "X-T30 II": "IV",
    "X-H2S": "V", "X-H2": "V", "X-T5": "V", "X-S20": "V", "X100VI": "V", "X-T50": "V", "X-M5": "V", "X-E5": "V", "X-T30 III": "V",
}

MODEL_NOTES = {
    "IV_EARLY": "Early X-Trans IV mode: X-T3 / X-T30 have fewer JPEG controls than later X-Trans IV bodies, so Clarity is removed and newer behaviour is kept conservative.",
    "V": "X-Trans V / Processor-5 generation mode: modern simulations and full JPEG controls are enabled where supported by the body/firmware.",
}

INTENT_PROFILES = {
    "Auto from image": {},
    "Natural / accurate": {"vintage": -0.10, "cinematic": -0.08},
    "Cinematic": {"cinematic": 0.18, "sat": -0.06, "contrast": 0.06},
    "Vintage film": {"vintage": 0.18, "warmth": 0.04, "sat": -0.04},
    "Soft portrait": {"softness": 0.16, "contrast": -0.08, "sat": -0.04},
    "Moody": {"cinematic": 0.12, "brightness": -0.12, "contrast": 0.10},
    "Vibrant travel": {"sat": 0.16, "contrast": 0.06, "brightness": 0.04},
    "Muted editorial": {"sat": -0.14, "cinematic": 0.08},
    "Warm lifestyle": {"warmth": 0.14, "softness": 0.06, "vintage": 0.08},
    "Black & white": {"low_colour": 0.35, "sat": -0.40, "contrast": 0.12},
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

def _sample_pixels(rgb: np.ndarray, max_pixels: int = 65000) -> np.ndarray:
    """Deterministic sampling for stable fingerprints without heavy computation."""
    flat = rgb.reshape(-1, 3)
    if flat.shape[0] <= max_pixels:
        return flat
    idx = np.linspace(0, flat.shape[0] - 1, max_pixels).astype(np.int64)
    return flat[idx]


def _hist(values: np.ndarray, bins: int, value_range: Tuple[float, float]) -> List[float]:
    h, _ = np.histogram(values, bins=bins, range=value_range)
    h = h.astype(np.float32)
    total = float(np.sum(h))
    if total <= 0:
        return [0.0] * bins
    return (h / total).tolist()


def _hist_similarity(a: List[float], b: List[float]) -> float:
    """Histogram-intersection similarity, 0-1."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(np.clip(np.sum(np.minimum(np.asarray(a), np.asarray(b))), 0, 1))


def _soft_hist_peak(center: float, bins: int, sigma: float = 0.16) -> List[float]:
    xs = np.linspace(0.0, 1.0, bins)
    vals = np.exp(-((xs - center) ** 2) / (2 * sigma * sigma))
    vals = vals / max(np.sum(vals), 1e-9)
    return vals.astype(float).tolist()


def _tone_hist_target(brightness: float, contrast: float) -> List[float]:
    bins = 10
    xs = np.linspace(0.0, 1.0, bins)
    sigma = float(np.clip(0.20 + (1.0 - contrast) * 0.12, 0.16, 0.34))
    core = np.exp(-((xs - brightness) ** 2) / (2 * sigma * sigma))
    if contrast > 0.68:
        core += (contrast - 0.68) * 1.4 * (
            np.exp(-((xs - 0.10) ** 2) / .018) +
            np.exp(-((xs - 0.90) ** 2) / .018)
        )
    core = core / max(np.sum(core), 1e-9)
    return core.astype(float).tolist()


def _fingerprint_from_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Compact look fingerprint inspired by film-stock fingerprinting workflows."""
    return {
        "tone_hist": features.get("tone_hist", []),
        "sat_hist": features.get("sat_hist", []),
        "hue_hist": features.get("hue_hist", []),
        "chroma_hist": features.get("chroma_hist", []),
        "brightness": float(features.get("brightness", .5)),
        "contrast": float(features.get("contrast", .5)),
        "warmth": float(features.get("warmth", .5)),
        "tint": float(features.get("tint", .5)),
        "sat": float(features.get("sat", .5)),
        "colorfulness": float(features.get("colorfulness", .5)),
        "shadow_warmth": float(features.get("shadow_warmth", .5)),
        "highlight_warmth": float(features.get("highlight_warmth", .5)),
        "grain_signature": float(features.get("grain_signature", .25)),
        "edge_density": float(features.get("edge_density", .1)),
        "palette_lab": features.get("palette_lab", []),
    }


def _grain_target_from_preset(preset: RecipePreset) -> float:
    grain = str(preset.grain_effect).lower()
    if "strong" in grain:
        return 0.78
    if "weak" in grain:
        return 0.48
    if "not available" in grain:
        return 0.22
    return 0.24


def _preset_fingerprint(preset: RecipePreset) -> Dict[str, Any]:
    t = preset.target
    film = preset.film_simulation
    fam = preset.family.lower()

    warmth = float(t.get("warmth", .5))
    sat = float(t.get("sat", .5))
    contrast = float(t.get("contrast", .5))
    brightness = float(t.get("brightness", .5))
    tint = float(t.get("tint", .5))
    colorfulness = float(np.clip(sat * 0.75 + (1.0 - t.get("softness", .5)) * 0.12, 0, 1))

    hue_bins = np.ones(12, dtype=np.float32) * 0.35
    if "Classic Negative" in film:
        hue_bins[[0, 1, 5, 6, 7]] += [0.60, 0.45, 0.38, 0.45, 0.34]
    elif "Classic Chrome" in film:
        hue_bins[[1, 2, 6, 7, 8]] += [0.45, 0.34, 0.35, 0.28, 0.22]
    elif "Nostalgic" in film:
        hue_bins[[0, 1, 2, 3]] += [0.48, 0.62, 0.34, 0.18]
    elif "Reala" in film or "Provia" in film:
        hue_bins += 0.18
    elif "Eterna" in film:
        hue_bins[[5, 6, 7, 8, 9]] += [0.24, 0.35, 0.42, 0.32, 0.20]
    elif "Velvia" in film:
        hue_bins[[0, 2, 4, 6, 8, 10]] += [0.48, 0.44, 0.50, 0.52, 0.46, 0.40]
    elif "Pro Neg" in film or "Astia" in film:
        hue_bins[[0, 1, 2]] += [0.34, 0.28, 0.18]
    elif "Acros" in film or "Monochrome" in film or "Sepia" in film:
        hue_bins *= 0.15
        if "Sepia" in film:
            hue_bins[[0, 1, 2]] += [0.5, 0.45, 0.25]
    hue_bins = (hue_bins / max(np.sum(hue_bins), 1e-9)).astype(float).tolist()

    shadow_warmth = warmth
    highlight_warmth = warmth
    if "cinematic" in fam or "Eterna" in film:
        shadow_warmth = float(np.clip(warmth - 0.08, 0, 1))
        highlight_warmth = float(np.clip(warmth + 0.03, 0, 1))
    if "nostalgic" in fam or "warm" in fam or "Kodachrome" in preset.name:
        highlight_warmth = float(np.clip(warmth + 0.08, 0, 1))
    if "cool" in fam:
        shadow_warmth = float(np.clip(warmth - 0.10, 0, 1))

    return {
        "tone_hist": _tone_hist_target(brightness, contrast),
        "sat_hist": _soft_hist_peak(sat, 8, sigma=0.18),
        "hue_hist": hue_bins,
        "chroma_hist": _soft_hist_peak(colorfulness, 8, sigma=0.20),
        "brightness": brightness,
        "contrast": contrast,
        "warmth": warmth,
        "tint": tint,
        "sat": sat,
        "colorfulness": colorfulness,
        "shadow_warmth": shadow_warmth,
        "highlight_warmth": highlight_warmth,
        "grain_signature": _grain_target_from_preset(preset),
        "edge_density": float(np.clip((preset.sharpness + preset.clarity + 8) / 16.0, 0, 1)),
    }


def fingerprint_similarity(features: Dict[str, Any], preset: RecipePreset) -> Tuple[float, Dict[str, float]]:
    img = features.get("fingerprint") or _fingerprint_from_features(features)
    tgt = _preset_fingerprint(preset)

    tone = _hist_similarity(img.get("tone_hist", []), tgt.get("tone_hist", []))
    sat_hist = _hist_similarity(img.get("sat_hist", []), tgt.get("sat_hist", []))
    hue = _hist_similarity(img.get("hue_hist", []), tgt.get("hue_hist", []))
    chroma = _hist_similarity(img.get("chroma_hist", []), tgt.get("chroma_hist", []))

    def gaussian(a, b, sigma):
        return math.exp(-((float(a) - float(b)) ** 2) / (2 * sigma * sigma))

    colour_cast = (
        gaussian(img.get("warmth", .5), tgt.get("warmth", .5), .18) * .34 +
        gaussian(img.get("tint", .5), tgt.get("tint", .5), .16) * .20 +
        gaussian(img.get("shadow_warmth", .5), tgt.get("shadow_warmth", .5), .20) * .22 +
        gaussian(img.get("highlight_warmth", .5), tgt.get("highlight_warmth", .5), .20) * .24
    )
    texture = (
        gaussian(img.get("grain_signature", .25), tgt.get("grain_signature", .25), .28) * .62 +
        gaussian(img.get("edge_density", .1), tgt.get("edge_density", .4), .34) * .38
    )

    score = (
        tone * 0.22 +
        hue * 0.17 +
        sat_hist * 0.15 +
        chroma * 0.11 +
        colour_cast * 0.24 +
        texture * 0.11
    )
    details = {
        "fp_tone": float(tone),
        "fp_hue": float(hue),
        "fp_saturation": float(sat_hist),
        "fp_chroma": float(chroma),
        "fp_colour_cast": float(colour_cast),
        "fp_texture": float(texture),
        "fp_score": float(np.clip(score, 0, 1)),
    }
    return details["fp_score"], details

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


def _colour_science_lab(rgb: np.ndarray) -> np.ndarray:
    """Return CIE Lab using colour-science when available.
    Output format is real CIE Lab: L* 0-100, a*/b* roughly -128..127.
    The function is intentionally defensive because colour-science APIs differ slightly by version.
    """
    if not HAS_COLOUR:
        raise RuntimeError("colour-science is not installed")
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    try:
        cs = colour.RGB_COLOURSPACES["sRGB"]
        # Convert encoded sRGB to linear RGB.
        try:
            linear = colour.cctf_decoding(arr, function="sRGB")
        except Exception:
            try:
                linear = colour.models.eotf_sRGB(arr)
            except Exception:
                linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
        # RGB_to_XYZ signatures changed across versions, so try the modern path first.
        try:
            XYZ = colour.RGB_to_XYZ(linear, cs.whitepoint, cs.whitepoint, cs.matrix_RGB_to_XYZ)
        except TypeError:
            XYZ = colour.RGB_to_XYZ(linear, colourspace=cs)
        Lab = colour.XYZ_to_Lab(XYZ, illuminant=cs.whitepoint)
        return np.asarray(Lab, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"colour-science conversion failed: {exc}")


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Return an OpenCV-style LAB array for the rest of the app.
    If colour-science is installed, this uses proper sRGB → XYZ → CIE Lab conversion,
    then maps it into the existing OpenCV-like scale: L 0..255, a/b centered at 128.
    """
    if HAS_COLOUR:
        try:
            Lab = _colour_science_lab(rgb)
            L = np.clip(Lab[:, :, 0] * 2.55, 0, 255)
            a = np.clip(Lab[:, :, 1] + 128.0, 0, 255)
            b = np.clip(Lab[:, :, 2] + 128.0, 0, 255)
            return np.stack([L, a, b], axis=2).astype(np.float32)
        except Exception:
            pass
    if HAS_CV2:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # Fallback approximation with PIL/NumPy if OpenCV is unavailable.
    arr = rgb.astype(np.float32) / 255.0
    L = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    a = arr[:, :, 0] - arr[:, :, 1]
    b = arr[:, :, 1] - arr[:, :, 2]
    return np.stack([L * 255.0, (a + 1) * 127.5, (b + 1) * 127.5], axis=2).astype(np.float32)


def delta_e_simple(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Perceptual colour difference helper. Uses Delta E 2000 if colour-science is available,
    otherwise falls back to Euclidean Lab distance. Inputs are real CIE Lab-like vectors.
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    if HAS_COLOUR:
        try:
            return float(colour.delta_E(lab1, lab2, method="CIE 2000"))
        except Exception:
            pass
    return float(np.linalg.norm(lab1 - lab2))


def kalmus_style_color_story(rgb: np.ndarray, lab_cv: np.ndarray, base: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight KALMUS-inspired single-image colour story analysis.
    KALMUS studies/visualises film colour usage; this recreates the useful idea without
    adding KALMUS as a heavy dependency. It produces a palette narrative, harmony estimate,
    colour diversity, and a compact colour-barcode strip for the UI/export.
    """
    arr = rgb.astype(np.float32) / 255.0
    flat = arr.reshape(-1, 3)
    if flat.shape[0] > 70000:
        idx = np.linspace(0, flat.shape[0] - 1, 70000).astype(int)
        flat = flat[idx]
    maxc = flat.max(axis=1)
    minc = flat.min(axis=1)
    sat = np.where(maxc <= 1e-6, 0, (maxc - minc) / np.maximum(maxc, 1e-6))
    # Warm/cool split uses both RGB channel tendency and blue/yellow behaviour.
    warm_mask = (flat[:, 0] > flat[:, 2] * 1.04) & (sat > 0.08)
    cool_mask = (flat[:, 2] > flat[:, 0] * 1.04) & (sat > 0.08)
    neutral_mask = sat <= 0.08
    warm_ratio = float(np.mean(warm_mask))
    cool_ratio = float(np.mean(cool_mask))
    neutral_ratio = float(np.mean(neutral_mask))
    # Hue distribution for diversity and harmony.
    if HAS_CV2:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
        if hsv.shape[0] > 70000:
            hsv = hsv[idx]
        hue = hsv[:, 0] / 180.0
        hsv_sat = hsv[:, 1] / 255.0
    else:
        R, G, B = flat[:, 0], flat[:, 1], flat[:, 2]
        hue = (np.arctan2(np.sqrt(3) * (G - B), 2 * R - G - B) + math.pi) / (2 * math.pi)
        hsv_sat = sat
    colour_pixels = hsv_sat > 0.12
    hue_hist = np.histogram(hue[colour_pixels] if np.any(colour_pixels) else hue, bins=12, range=(0, 1))[0].astype(float)
    if hue_hist.sum() > 0:
        hue_hist /= hue_hist.sum()
    entropy = -float(np.sum(hue_hist * np.log2(hue_hist + 1e-9))) / math.log2(12)
    top_bins = np.argsort(hue_hist)[-3:][::-1]
    spread = float((top_bins[0] - top_bins[-1]) % 12) / 12.0 if len(top_bins) >= 3 else 0.0
    mono_strength = float(np.max(hue_hist)) if hue_hist.size else 0.0
    analogous_strength = float(max(sum(hue_hist[i:i+3]) if i <= 9 else sum(np.r_[hue_hist[i:], hue_hist[:(i+3)%12]]) for i in range(12))) if hue_hist.size else 0.0
    complementary_strength = float(max(hue_hist[i] + hue_hist[(i + 6) % 12] for i in range(12))) if hue_hist.size else 0.0
    if mono_strength > 0.62 or neutral_ratio > 0.55:
        harmony = "monochrome / restrained"
    elif analogous_strength > 0.68:
        harmony = "analogous / cohesive"
    elif complementary_strength > 0.48 and entropy > 0.45:
        harmony = "complementary contrast"
    elif entropy > 0.72:
        harmony = "varied / travel colour"
    else:
        harmony = "balanced palette"
    # Quantised colour barcode: sorted by luminance to give a film-strip-like readout.
    sample_u8 = np.clip(flat * 255, 0, 255).astype(np.uint8)
    q = (sample_u8 // 32).astype(np.int32)
    packed = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts)[-10:][::-1]
    barcode = []
    for v in vals[order]:
        rr = v // 64
        gg = (v % 64) // 8
        bb = v % 8
        center = np.clip(np.array([rr, gg, bb]) * 32 + 16, 0, 255).astype(int)
        barcode.append("#%02x%02x%02x" % tuple(center))
    # Palette separation using Lab centres from the barcode colours.
    lab_centres = []
    for hx in barcode[:6]:
        rgb_c = np.array([[[int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16)]]], dtype=np.uint8)
        lab_c = rgb_to_lab(rgb_c)[0, 0]
        lab_centres.append(np.array([lab_c[0]/2.55, lab_c[1]-128, lab_c[2]-128], dtype=float))
    if len(lab_centres) > 1:
        dists = []
        for i in range(len(lab_centres)):
            for j in range(i+1, len(lab_centres)):
                dists.append(delta_e_simple(lab_centres[i], lab_centres[j]))
        palette_contrast = float(np.clip(np.mean(dists) / 55.0, 0, 1))
    else:
        palette_contrast = 0.0
    if base.get("low_colour", 0) > .70:
        mood = "monochrome / tonal study"
    elif warm_ratio > cool_ratio + .18 and base.get("sat", .5) < .42:
        mood = "warm muted print"
    elif cool_ratio > warm_ratio + .16 and base.get("contrast", .5) > .52:
        mood = "cool cinematic"
    elif base.get("sat", .5) > .62 and palette_contrast > .45:
        mood = "vibrant travel colour"
    elif base.get("softness", .5) > .65 and base.get("sat", .5) < .45:
        mood = "soft editorial"
    else:
        mood = "natural balanced colour"
    story_score = float(np.clip((entropy * .25) + (palette_contrast * .25) + ((1-neutral_ratio) * .20) + (abs(warm_ratio-cool_ratio) * .30), 0, 1))
    return {
        "engine": "colour-science" if HAS_COLOUR else "opencv/numpy fallback",
        "colour_engine_active": bool(HAS_COLOUR),
        "mood": mood,
        "harmony": harmony,
        "warm_ratio": warm_ratio,
        "cool_ratio": cool_ratio,
        "neutral_ratio": neutral_ratio,
        "diversity": float(np.clip(entropy, 0, 1)),
        "palette_contrast": palette_contrast,
        "dominant_strength": mono_strength,
        "analogous_strength": analogous_strength,
        "complementary_strength": complementary_strength,
        "story_score": story_score,
        "barcode": barcode,
        "barcode_css": "linear-gradient(90deg, " + ", ".join(barcode or ["#111111"]) + ")",
    }


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
        blur = cv2.GaussianBlur(gray, (0, 0), 1.6)
        grain_signature = float(np.clip(np.std(gray - blur) / 28.0, 0, 1))
    else:
        gray = np.mean(arr, axis=2)
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        grad = np.sqrt(gx * gx + gy * gy)
        sharpness = float(np.clip(np.var(grad) * 25, 0, 1))
        edge_density = float(np.clip(np.mean(grad > np.percentile(grad, 85)), 0, 1))
        grain_signature = float(np.clip(np.std(grad) * 8.0, 0, 1))

    softness = float(np.clip((softness * .70) + ((1 - sharpness) * .30), 0, 1))

    # Cinematic and vintage are style estimates, not hard truth.
    cinematic = float(np.clip((contrast * .35) + ((1 - saturation) * .35) + ((1 - l_mean) * .20) + (edge_density * .10), 0, 1))
    vintage = float(np.clip((low_colour * .25) + (abs(warmth - .5) * .55) + ((1 - saturation) * .20) + (contrast * .15), 0, 1))

    # Fingerprint histograms: stable, compact, and useful for comparing visual behaviour.
    if HAS_CV2:
        hsv_for_fp = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        H_fp = hsv_for_fp[:, :, 0].astype(np.float32) / 180.0
        S_fp = hsv_for_fp[:, :, 1].astype(np.float32) / 255.0
    else:
        H_fp = (np.arctan2(np.sqrt(3) * (G - B), 2 * R - G - B) + math.pi) / (2 * math.pi)
        S_fp = sat.astype(np.float32)
    tone_hist = _hist(L.flatten() / 255.0, 10, (0.0, 1.0))
    sat_hist = _hist(S_fp.flatten(), 8, (0.0, 1.0))
    hue_hist = _hist(H_fp.flatten(), 12, (0.0, 1.0))
    chroma_hist = _hist(np.clip(chroma / 90.0, 0, 1).flatten(), 8, (0.0, 1.0))

    sample = _sample_pixels(rgb, 55000)
    q = (sample // 32).astype(np.int32)
    packed = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts)[-6:][::-1]
    palette_lab = []
    for val, cnt in zip(vals[order], counts[order]):
        rr = val // 64
        gg = (val % 64) // 8
        bb = val % 8
        rgb_center = np.array([[[rr * 32 + 16, gg * 32 + 16, bb * 32 + 16]]], dtype=np.uint8)
        lab_center = rgb_to_lab(rgb_center)[0, 0]
        palette_lab.append({"lab": [float(lab_center[0] / 255.0), float((lab_center[1]-128)/128.0), float((lab_center[2]-128)/128.0)], "weight": float(cnt / max(np.sum(counts[order]), 1))})

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

    # KALMUS-inspired colour-story layer: palette harmony, diversity, warm/cool balance,
    # and a colour-barcode readout. This improves explanation quality and adds a
    # lightweight perceptual colour score without making KALMUS a dependency.
    color_story = kalmus_style_color_story(rgb, lab, {
        "warmth": warmth, "tint": tint, "sat": saturation, "contrast": contrast,
        "brightness": l_mean, "softness": softness, "cinematic": cinematic,
        "vintage": vintage, "low_colour": low_colour
    })

    zone_data = detect_zones_and_scene(rgb, {
        "warmth": warmth, "tint": tint, "sat": saturation, "contrast": contrast,
        "brightness": l_mean, "softness": softness, "cinematic": cinematic,
        "vintage": vintage, "low_colour": low_colour
    })

    process_data = detect_film_process(rgb, L, {
        "warmth": warmth, "tint": tint, "sat": saturation, "contrast": contrast,
        "brightness": l_mean, "softness": softness, "sharpness": sharpness,
        "colorfulness": colorfulness, "vintage": vintage, "edge_density": edge_density,
        "grain_signature": grain_signature, "shadow_warmth": shadow_warmth,
        "highlight_warmth": highlight_warmth, **zone_data
    })

    result_features = {
        "warmth": warmth,
        "tint": tint,
        "sat": saturation,
        "colorfulness": colorfulness,
        "contrast": contrast,
        "brightness": l_mean,
        "softness": softness,
        "sharpness": sharpness,
        "edge_density": edge_density,
        "grain_signature": grain_signature,
        "tone_hist": tone_hist,
        "sat_hist": sat_hist,
        "hue_hist": hue_hist,
        "chroma_hist": chroma_hist,
        "palette_lab": palette_lab,
        "fingerprint": None,
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
        "color_story": color_story,
        "colour_engine_active": color_story.get("colour_engine_active", False),
        "color_story_mood": color_story.get("mood", ""),
        "color_story_harmony": color_story.get("harmony", ""),
        **zone_data,
        **process_data,
    }
    result_features["fingerprint"] = _fingerprint_from_features(result_features)
    return result_features


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


def detect_zones_and_scene(rgb: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
    """Detect semantic colour zones without heavy ML dependencies.
    This is not object detection; it is a practical photography-oriented heuristic layer.
    """
    arr = rgb.astype(np.float32) / 255.0
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mx = np.max(arr, axis=2)
    mn = np.min(arr, axis=2)
    sat = np.where(mx <= 1e-6, 0, (mx - mn) / np.maximum(mx, 1e-6))
    # HSV hue via cv2 if available for more stable masks.
    if HAS_CV2:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        H, S, V = hsv[:, :, 0].astype(np.float32), hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
        skin = ((H <= 25) | (H >= 170)) & (S >= 25) & (S <= 190) & (V >= 45)
        sky = (H >= 90) & (H <= 130) & (S >= 25) & (V >= 70)
        foliage = (H >= 35) & (H <= 85) & (S >= 25) & (V >= 35)
        warm_food = ((H <= 28) | (H >= 165)) & (S >= 35) & (V >= 35)
        neon = ((S > 140) & (V > 100) & ((H < 15) | (H > 135) | ((H > 85) & (H < 125))))
    else:
        skin = (R > .30) & (G > .18) & (B > .12) & (R > B) & ((R - G) < .22) & (sat > .12)
        sky = (B > R * 1.10) & (B > G * .95) & (sat > .15)
        foliage = (G > R * 1.05) & (G > B * 1.05) & (sat > .12)
        warm_food = (R > G * .95) & (G > B * 1.05) & (sat > .18)
        neon = (sat > .60) & (mx > .55)
    dark = mx < .25
    bright = mx > .78
    total = float(rgb.shape[0] * rgb.shape[1])
    ratios = {
        "skin_ratio": float(np.sum(skin) / total),
        "sky_ratio": float(np.sum(sky) / total),
        "foliage_ratio": float(np.sum(foliage) / total),
        "warm_object_ratio": float(np.sum(warm_food) / total),
        "neon_ratio": float(np.sum(neon) / total),
        "dark_ratio": float(np.sum(dark) / total),
        "bright_ratio": float(np.sum(bright) / total),
    }
    scene = "general"
    confidence = 0.40
    if features.get("low_colour", 0) > .72:
        scene, confidence = "black & white", .82
    elif ratios["neon_ratio"] > .035 or (ratios["dark_ratio"] > .48 and features.get("contrast", 0) > .55):
        scene, confidence = "night / neon", min(.90, .55 + ratios["neon_ratio"] * 3)
    elif ratios["skin_ratio"] > .10:
        scene, confidence = "portrait", min(.90, .50 + ratios["skin_ratio"] * 2.2)
    elif ratios["sky_ratio"] > .12 or ratios["foliage_ratio"] > .18:
        scene, confidence = "landscape", min(.88, .45 + ratios["sky_ratio"] + ratios["foliage_ratio"] * 1.2)
    elif ratios["warm_object_ratio"] > .18 and features.get("warmth", .5) > .54:
        scene, confidence = "cafe / food / product", min(.84, .45 + ratios["warm_object_ratio"])
    elif features.get("cinematic", .5) > .65 or features.get("vintage", .5) > .65:
        scene, confidence = "street / documentary", .58
    ratios["scene"] = scene
    ratios["scene_confidence"] = float(confidence)
    return ratios


def detect_film_process(rgb: np.ndarray, L: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight analog-film process detector inspired by spectral film pipelines.

    This does not simulate film. It detects visual cues that usually come from
    analog scans or film emulation: glow/halation, lifted blacks, print warmth,
    dye-like muted colour, split colour-casts, and grain/scan texture.
    """
    arr = rgb.astype(np.float32) / 255.0
    lum = np.clip(L.astype(np.float32) / 255.0, 0, 1)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    p05, p50, p95 = [float(np.percentile(lum, q)) for q in [5, 50, 95]]
    bright = lum > max(0.78, p95)

    if HAS_CV2 and np.any(bright):
        bright_u8 = bright.astype(np.uint8) * 255
        kernel = np.ones((9, 9), np.uint8)
        halo_ring = (cv2.dilate(bright_u8, kernel, iterations=2) > 0) & (~bright)
        halo_lift = float(np.mean(lum[halo_ring]) - np.mean(lum[~bright])) if np.any(halo_ring) and np.any(~bright) else 0.0
        warm_halo = float(np.mean((R - B)[halo_ring])) if np.any(halo_ring) else 0.0
        halation = float(np.clip((halo_lift * 2.8) + max(0.0, warm_halo) * 1.2 + features.get("bright_ratio", 0) * 0.35, 0, 1))
    else:
        halation = float(np.clip(features.get("bright_ratio", 0) * features.get("softness", 0.5) * 1.8, 0, 1))

    bloom = float(np.clip(features.get("bright_ratio", 0) * 1.2 + features.get("softness", 0.5) * 0.45 - features.get("sharpness", 0.5) * 0.25, 0, 1))
    matte_black = float(np.clip((p05 - 0.025) * 7.0 + (1.0 - features.get("contrast", 0.5)) * 0.25, 0, 1))
    highlight_warmth = features.get("highlight_warmth", 0.5)
    shadow_warmth = features.get("shadow_warmth", 0.5)
    print_warmth = float(np.clip((highlight_warmth - 0.5) * 1.8 + features.get("warmth", 0.5) * 0.65, 0, 1))
    negative_density = float(np.clip((1.0 - abs(p50 - 0.43) * 2.2) * 0.35 + features.get("contrast", 0.5) * 0.35 + features.get("colorfulness", 0.4) * 0.30, 0, 1))
    dye_muting = float(np.clip((1.0 - abs(features.get("sat", 0.5) - 0.38) * 2.4) * 0.55 + features.get("vintage", 0.4) * 0.30 + matte_black * 0.15, 0, 1))
    colour_split = float(np.clip(abs(highlight_warmth - shadow_warmth) * 2.2, 0, 1))
    film_grain = float(np.clip(features.get("grain_signature", 0) * 0.85 + max(0.0, features.get("edge_density", 0) - 0.06) * 0.9, 0, 1))
    analog_character = float(np.clip(halation * 0.20 + bloom * 0.16 + matte_black * 0.17 + print_warmth * 0.14 + dye_muting * 0.14 + colour_split * 0.10 + film_grain * 0.09, 0, 1))

    flags = []
    if halation > 0.45:
        flags.append("halation / warm glow")
    if bloom > 0.55:
        flags.append("bloom or diffusion")
    if matte_black > 0.45:
        flags.append("lifted matte blacks")
    if print_warmth > 0.62:
        flags.append("warm print-paper bias")
    if dye_muting > 0.58:
        flags.append("muted dye-like colour")
    if colour_split > 0.45:
        flags.append("warm/cool split toning")
    if film_grain > 0.55:
        flags.append("visible grain / scan texture")

    return {
        "film_process": {
            "halation": halation,
            "bloom": bloom,
            "matte_black": matte_black,
            "print_warmth": print_warmth,
            "negative_density": negative_density,
            "dye_muting": dye_muting,
            "colour_split": colour_split,
            "film_grain": film_grain,
            "analog_character": analog_character,
            "flags": flags,
        },
        "halation_score": halation,
        "bloom_score": bloom,
        "matte_black_score": matte_black,
        "analog_character": analog_character,
    }


def apply_intent(features: Dict[str, Any], intent: str) -> Dict[str, Any]:
    adjusted = dict(features)
    for key, delta in INTENT_PROFILES.get(intent, {}).items():
        if isinstance(adjusted.get(key), (int, float)):
            adjusted[key] = float(np.clip(adjusted[key] + delta, 0, 1))
    adjusted["intent"] = intent
    return adjusted


def camera_to_sensor(camera_model: str, fallback: str) -> Tuple[str, str]:
    mapped = CAMERA_MODEL_MAP.get(camera_model)
    if not mapped:
        return fallback, ""
    return mapped, MODEL_NOTES.get(mapped, "")


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
    if sensor_code == "IV_EARLY":
        return "Early X-Trans IV mode for X-T3 / X-T30: Clarity and Color Chrome FX Blue are hidden; later X-Trans IV settings are simplified."
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

    if sensor_code == "IV_EARLY":
        r["clarity"] = "Not available"
        r["color_chrome_fx_blue"] = "Not available"
        r["grain_effect"] = simplify_grain_for_sensor(r.get("grain_effect", "Off"), "III")

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

    # Scene-aware nudges: same colour data should be interpreted differently for portraits, landscapes, night, etc.
    scene = features.get("scene", "general")
    film = preset.film_simulation
    fam = preset.family.lower()
    if scene == "portrait":
        if film in ["Pro Neg. Std", "Astia / Soft", "Reala Ace", "Nostalgic Negative"]:
            score += .14 * total
        if film == "Velvia / Vivid":
            score -= .16 * total
    elif scene == "landscape":
        if film in ["Velvia / Vivid", "Reala Ace", "Classic Chrome"]:
            score += .12 * total
        if "soft portrait" in fam:
            score -= .08 * total
    elif scene == "cafe / food / product":
        if film in ["Reala Ace", "Classic Chrome", "Nostalgic Negative", "Astia / Soft"]:
            score += .12 * total
    elif scene == "night / neon":
        if film in ["Classic Negative", "Eterna / Cinema", "Eterna Bleach Bypass"]:
            score += .16 * total
        if film in ["Velvia / Vivid", "Astia / Soft"]:
            score -= .10 * total
    elif scene == "black & white":
        if is_bw_preset:
            score += .18 * total
        else:
            score -= .25 * total
    elif scene == "street / documentary":
        if film in ["Classic Chrome", "Classic Negative", "Acros", "Acros + R Filter"]:
            score += .10 * total

    # NegClone-inspired fingerprint layer: compare tone, hue distribution, chroma,
    # split colour-cast and texture signature. This makes the recommender less dependent
    # on a few whole-image averages.
    fp_score, fp_details = fingerprint_similarity(features, preset)
    details.update(fp_details)
    fp_weight = 2.10
    score += fp_score * fp_weight
    total += fp_weight

    # Colour-science / KALMUS-style colour story score. This is not another
    # exact-match claim; it checks whether the palette behaviour supports the
    # suggested recipe family.
    story = features.get("color_story", {}) or {}
    preset_warm = float(preset.target.get("warmth", .5))
    preset_sat = float(preset.target.get("sat", .5))
    preset_contrast = float(preset.target.get("contrast", .5))
    warm_balance = float(story.get("warm_ratio", .33)) - float(story.get("cool_ratio", .33))
    story_warm = float(np.clip(.5 + warm_balance, 0, 1))
    story_sat = float(np.clip((1 - story.get("neutral_ratio", .3)) * .75 + story.get("diversity", .5) * .25, 0, 1))
    story_contrast = float(story.get("palette_contrast", .5))
    colour_story_score = (
        math.exp(-((story_warm - preset_warm) ** 2) / (2 * .26 * .26)) * .38 +
        math.exp(-((story_sat - preset_sat) ** 2) / (2 * .30 * .30)) * .30 +
        math.exp(-((story_contrast - preset_contrast) ** 2) / (2 * .30 * .30)) * .22 +
        float(story.get("story_score", .5)) * .10
    )
    details["color_story_score"] = float(np.clip(colour_story_score, 0, 1))
    story_weight = 0.90
    score += details["color_story_score"] * story_weight
    total += story_weight

    final = float(np.clip(score / total, 0, 1))
    return final, details


def recommend(features: Dict[str, Any], sensor_code: str, intent: str = "Auto from image") -> Dict[str, Any]:
    raw_features = dict(features)
    features = apply_intent(features, intent)
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
        "fingerprint_fit": best["details"].get("fp_score", 0.0),
        "match_details": best["details"],
        "alternatives": [{"name": c["preset"].name, "film_simulation": c["preset"].film_simulation, "fit": c["score"], "fingerprint_fit": c["details"].get("fp_score", 0.0), "best_for": c["preset"].best_for} for c in alts],
        "visual_summary": summary,
        "guidance": guidance,
        "features": features,
        "raw_features": raw_features,
        "intent": intent,
        "scene": features.get("scene", "general"),
        "scene_confidence": features.get("scene_confidence", 0.0),
        "variants": build_recipe_variants(recipe, sensor_code),
        "realism": in_camera_realism(features, recipe),
    }


def _fuji_int(value: Any, fallback: int = 0) -> int:
    """Safely coerce Fuji recipe controls to int."""
    try:
        return int(value)
    except Exception:
        return fallback


def _clamp_recipe_control(recipe: Dict[str, Any], key: str, lo: int, hi: int) -> None:
    recipe[key] = int(np.clip(_fuji_int(recipe.get(key, 0)), lo, hi))


def apply_fuji_safe_tone_color_guardrails(recipe: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Conservative Fuji-JPEG guardrails.

    Earlier versions pushed Highlight / Shadow / Color / WB too directly from image
    features. That creates the common failure mode users reported: crushed shadows,
    clipped highlights, and colour casts that feel wrong. This function intentionally
    keeps the generated recipe in a safer working range unless the reference image
    very clearly demands a stronger look.
    """
    r = dict(recipe)

    brightness = float(features.get("brightness", 0.5))
    contrast = float(features.get("contrast", 0.5))
    sat = float(features.get("sat", 0.5))
    colorfulness = float(features.get("colorfulness", sat))
    low_colour = float(features.get("low_colour", 0.0))
    dark_ratio = float(features.get("dark_ratio", 0.0))
    highlight_ratio = float(features.get("highlight_ratio", 0.0))
    p5 = float(features.get("luminance_p5", 0.08))
    p95 = float(features.get("luminance_p95", 0.88))
    scene = str(features.get("scene", "general"))

    # 1) Tone controls: keep safer than old versions.
    # Fuji Highlight/Shadow controls are not general contrast sliders; too much Shadow
    # quickly destroys shadow detail, and too much Highlight can make skin/skies harsh.
    _clamp_recipe_control(r, "highlights", -2, 1)
    _clamp_recipe_control(r, "shadows", -1, 2)

    # Protect highlights if the reference is high-key, has bright whites/sky, or broad DR.
    if brightness > 0.66 or p95 > 0.88 or highlight_ratio > 0.08:
        r["highlights"] = min(_fuji_int(r.get("highlights")), -1)
    if brightness > 0.76 or highlight_ratio > 0.16:
        r["highlights"] = min(_fuji_int(r.get("highlights")), -2)

    # Protect shadows. The app should not crush blacks just because a reference is moody.
    if dark_ratio > 0.34 or p5 < 0.045:
        r["shadows"] = min(_fuji_int(r.get("shadows")), 1)
    if dark_ratio > 0.48 or p5 < 0.025:
        r["shadows"] = min(_fuji_int(r.get("shadows")), 0)

    # Low-contrast images should get gentler shadows, not darker shadows.
    if contrast < 0.42:
        r["shadows"] = min(_fuji_int(r.get("shadows")), 0)
        r["highlights"] = min(_fuji_int(r.get("highlights")), -1)

    # Very contrasty references: allow some shadow punch, but do not exceed +2.
    if contrast > 0.76 and dark_ratio < 0.30:
        r["shadows"] = min(max(_fuji_int(r.get("shadows")), 1), 2)

    # Portraits and lifestyle/cafe images are punished heavily by harsh tone settings.
    if scene in {"portrait", "cafe / food / product"}:
        r["highlights"] = min(_fuji_int(r.get("highlights")), -1)
        r["shadows"] = min(_fuji_int(r.get("shadows")), 1)

    # 2) Colour guardrails. Avoid wildly saturated or desaturated recommendations.
    _clamp_recipe_control(r, "color", -2, 2)
    if low_colour > 0.58 or sat < 0.26 or colorfulness < 0.24:
        r["color"] = min(_fuji_int(r.get("color")), 0)
    elif sat > 0.72 and colorfulness > 0.55:
        r["color"] = max(_fuji_int(r.get("color")), 1)
    else:
        # Most real-world references sit best around -1/0/+1.
        r["color"] = int(np.clip(_fuji_int(r.get("color")), -1, 1))

    # 3) WB shift guardrails. The previous versions could drift too warm/green/magenta.
    # Keep most auto-generated recipes within practical Fuji ranges unless strong evidence.
    r["wb_shift_r"] = int(np.clip(_fuji_int(r.get("wb_shift_r")), -4, 4))
    r["wb_shift_b"] = int(np.clip(_fuji_int(r.get("wb_shift_b")), -5, 4))

    warmth = float(features.get("warmth", 0.5))
    tint = float(features.get("tint", 0.5))
    if 0.42 <= warmth <= 0.62:
        # Neutral references should not receive strong warm/cool bias.
        r["wb_shift_r"] = int(np.clip(r["wb_shift_r"], -2, 2))
        r["wb_shift_b"] = int(np.clip(r["wb_shift_b"], -3, 2))
    if 0.45 <= tint <= 0.56:
        # Avoid unnecessary magenta/green correction when tint is not clearly biased.
        r["wb_shift_r"] = int(np.clip(r["wb_shift_r"], -3, 3))

    # 4) Dynamic Range should protect capture, not mimic final contrast.
    drange = p95 - p5
    if highlight_ratio > 0.10 or drange > 0.66 or scene in {"portrait", "landscape", "cafe / food / product"}:
        r["dynamic_range"] = "DR400"
    elif drange > 0.48:
        r["dynamic_range"] = "DR200"
    else:
        r["dynamic_range"] = "DR200"  # safer default than DR100 for unknown references

    # 5) Exposure comp: keep conservative. Big changes are scene/exposure dependent.
    if brightness > 0.74 or highlight_ratio > 0.14:
        r["exposure_comp"] = "0"
    elif brightness < 0.30 and dark_ratio > 0.42:
        r["exposure_comp"] = "0"
    elif brightness < 0.40 and highlight_ratio < 0.04:
        r["exposure_comp"] = "+1/3"
    else:
        r["exposure_comp"] = "0"

    # 6) Detail controls: avoid brittle digital results.
    _clamp_recipe_control(r, "sharpness", -2, 1)
    _clamp_recipe_control(r, "noise_reduction", -4, -1)
    _clamp_recipe_control(r, "clarity", -2, 2)

    return r


def tune_recipe_from_features(p: RecipePreset, features: Dict[str, Any]) -> Dict[str, Any]:
    recipe = asdict(p)
    recipe.pop("target", None)

    # Conservative generator v5: use the preset as a Fuji-safe base and make only
    # small, evidence-driven changes. The goal is a usable starting recipe, not
    # a mathematically overfit response to one edited reference image.
    warmth = float(features.get("warmth", .5))
    tint = float(features.get("tint", .5))
    sat = float(features.get("sat", .5))
    contrast = float(features.get("contrast", .5))
    brightness = float(features.get("brightness", .5))
    softness = float(features.get("softness", .5))
    low_colour = float(features.get("low_colour", .0))
    dark_ratio = float(features.get("dark_ratio", .0))
    highlight_ratio = float(features.get("highlight_ratio", .0))

    # WB: only shift if the reference is clearly warm/cool/tinted. This prevents
    # common failures where skin and neutrals become orange, cyan, green, or magenta.
    if warmth > .74:
        recipe["wb_shift_r"] = _fuji_int(recipe["wb_shift_r"]) + 1
        recipe["wb_shift_b"] = _fuji_int(recipe["wb_shift_b"]) - 1
    elif warmth < .28:
        recipe["wb_shift_r"] = _fuji_int(recipe["wb_shift_r"]) - 1
        recipe["wb_shift_b"] = _fuji_int(recipe["wb_shift_b"]) + 1

    if tint < .36:  # clearly green leaning reference
        recipe["wb_shift_r"] = _fuji_int(recipe["wb_shift_r"]) - 1
    elif tint > .64:  # clearly magenta leaning reference
        recipe["wb_shift_r"] = _fuji_int(recipe["wb_shift_r"]) + 1

    # Color: small changes only. Fuji Color +2/+3 can become very wrong across scenes.
    if sat > .76 and low_colour < .22:
        recipe["color"] = _fuji_int(recipe["color"]) + 1
    elif sat < .26 or low_colour > .70:
        recipe["color"] = _fuji_int(recipe["color"]) - 1

    # Tone: respond to contrast carefully. Highlight protection comes later.
    if contrast > .80 and dark_ratio < .32:
        recipe["shadows"] = _fuji_int(recipe["shadows"]) + 1
    elif contrast < .36:
        recipe["shadows"] = _fuji_int(recipe["shadows"]) - 1
        recipe["highlights"] = _fuji_int(recipe["highlights"]) - 1

    # Brightness: never brighten by crushing highlights; use DR/highlight protection.
    if brightness > .68 or highlight_ratio > .10:
        recipe["highlights"] = _fuji_int(recipe["highlights"]) - 1
    elif brightness < .30 and dark_ratio < .35:
        recipe["shadows"] = _fuji_int(recipe["shadows"]) + 1

    # Detail: more conservative than previous version.
    if softness > .74:
        recipe["clarity"] = _fuji_int(recipe["clarity"]) - 1
        recipe["sharpness"] = _fuji_int(recipe["sharpness"]) - 1
    elif softness < .24:
        recipe["clarity"] = _fuji_int(recipe["clarity"]) + 1

    return apply_fuji_safe_tone_color_guardrails(recipe, features)

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

    process = features.get("film_process", {})
    flags = process.get("flags", [])
    if flags:
        lines.append("Analog-process cues detected: " + ", ".join(flags[:4]) + ". Treat the recipe as a camera base, not the full final grade.")
    if process.get("halation", 0) > .45 or process.get("bloom", 0) > .55:
        lines.append("Glow/halation is likely coming from lens diffusion, highlights, or editing. A Fuji recipe can set the colour base, but not create real bloom by itself.")
    if process.get("matte_black", 0) > .45:
        lines.append("Lifted blacks detected: keep shadows gentle, but expect matte fade to need exposure choice or light post-processing.")

    story = features.get("color_story", {}) or {}
    if story:
        lines.append(f"Colour-story read: {story.get('mood', 'balanced colour')} with {story.get('harmony', 'balanced palette')} harmony. Use this as the creative reason behind the film-simulation direction, not as an exact recipe ID.")
        if story.get("colour_engine_active"):
            lines.append("Colour Science Engine is active: palette/tint readings use sRGB → XYZ → CIE Lab conversion for more perceptual colour judgement.")

    return lines




def _safe_int_delta(value, delta, lo=-4, hi=4):
    if not isinstance(value, (int, np.integer)):
        return value
    return int(np.clip(int(value) + delta, lo, hi))


def build_recipe_variants(recipe: Dict[str, Any], sensor_code: str) -> List[Dict[str, Any]]:
    """Create practical subtle / balanced / strong versions of the same direction."""
    variants = []
    profiles = [
        ("Subtle", {"color": -1, "shadows": -1, "highlights": 0, "clarity": -1, "sharpness": -1, "wb_shift_scale": 0.75}),
        ("Balanced", {"wb_shift_scale": 1.0}),
        ("Strong", {"color": 1, "shadows": 1, "clarity": 1, "wb_shift_scale": 1.15}),
    ]
    for label, delta in profiles:
        r = dict(recipe)
        r["variant"] = label
        r["name"] = f"{recipe.get('name', 'Fuji Look')} — {label}"
        for k in ["color", "shadows", "highlights", "clarity", "sharpness"]:
            if k in delta:
                r[k] = _safe_int_delta(r.get(k), delta[k], -5 if k == "clarity" else -4, 5 if k == "clarity" else 4)
        scale = delta.get("wb_shift_scale", 1.0)
        if isinstance(r.get("wb_shift_r"), (int, np.integer)):
            r["wb_shift_r"] = int(np.clip(round(r["wb_shift_r"] * scale), -9, 9))
        if isinstance(r.get("wb_shift_b"), (int, np.integer)):
            r["wb_shift_b"] = int(np.clip(round(r["wb_shift_b"] * scale), -9, 9))
        variants.append(sanitize_recipe_for_sensor(r, sensor_code))
    return variants


def in_camera_realism(features: Dict[str, Any], recipe: Dict[str, Any]) -> Dict[str, Any]:
    warnings = []
    score = 0.86
    process = features.get("film_process", {})

    if features.get("neon_ratio", 0) > .05:
        warnings.append("Strong neon or mixed artificial light may need post-processing for glow/halation.")
        score -= .12
    if features.get("contrast", .5) > .84:
        warnings.append("Very hard contrast may require careful exposure; JPEG tone controls cannot fully replace local masking.")
        score -= .08
    if features.get("softness", .5) > .82:
        warnings.append("Heavy haze, bloom or diffusion may require lens/filter/light conditions, not just a recipe.")
        score -= .08
    if features.get("sat", .5) < .18 and features.get("low_colour", 0) < .65:
        warnings.append("Selective desaturation or editorial grading may not be fully achievable in-camera.")
        score -= .07
    if features.get("highlight_warmth", .5) - features.get("shadow_warmth", .5) > .22:
        warnings.append("Warm highlights plus cool shadows suggests split toning; Fuji recipes can approximate but not fully isolate tones.")
        score -= .08

    if process.get("halation", 0) > .45:
        warnings.append("Halation / warm glow detected. In-camera recipes cannot truly generate red highlight bloom; use lighting, diffusion, or editing if needed.")
        score -= .10
    if process.get("bloom", 0) > .58:
        warnings.append("Bloom/diffusion cues detected. The recipe can approximate softness, but optical glow needs a filter, lens behaviour, or post-processing.")
        score -= .08
    if process.get("matte_black", 0) > .50:
        warnings.append("Lifted matte blacks detected. Fuji shadow controls can soften contrast, but true matte fade may need editing or careful exposure.")
        score -= .06
    if process.get("colour_split", 0) > .50:
        warnings.append("Strong warm/cool tonal split detected. JPEG WB shift affects the whole frame, so isolated split-toning may need editing.")
        score -= .07
    if process.get("film_grain", 0) > .62:
        warnings.append("Visible grain/scan texture detected. Fuji grain helps, but scan texture or heavy film grain may not match exactly in-camera.")
        score -= .05

    if not warnings:
        warnings.append("This look is reasonably achievable as an in-camera JPEG starting point if exposure and lighting are close.")
    score = float(np.clip(score, .38, .92))
    if score >= .80:
        label = "Mostly achievable in-camera"
    elif score >= .65:
        label = "Achievable with careful shooting"
    else:
        label = "Recipe gets close, editing likely needed"
    return {"label": label, "score": score, "warnings": warnings, "film_process": process}


def compact_recipe_table(recipe: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [
        ("Film Simulation", str(recipe.get("film_simulation"))),
        ("Dynamic Range", str(recipe.get("dynamic_range"))),
        ("White Balance", str(recipe.get("white_balance"))),
        ("WB Shift", f"R {format_signed(recipe.get('wb_shift_r'))}, B {format_signed(recipe.get('wb_shift_b'))}"),
        ("Highlight", format_signed(recipe.get("highlights"))),
        ("Shadow", format_signed(recipe.get("shadows"))),
        ("Color", format_signed(recipe.get("color"))),
        ("Sharpness", format_signed(recipe.get("sharpness"))),
        ("Noise Reduction", format_signed(recipe.get("noise_reduction"))),
        ("Clarity", format_signed(recipe.get("clarity"))),
        ("Grain", str(recipe.get("grain_effect"))),
        ("Color Chrome", str(recipe.get("color_chrome_effect"))),
        ("FX Blue", str(recipe.get("color_chrome_fx_blue"))),
        ("ISO", str(recipe.get("iso"))),
        ("Exposure Comp", str(recipe.get("exposure_comp"))),
    ]

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
        story = result.get("features", {}).get("color_story", {}) if result else {}
        if story:
            lines.append(f"Color Story          : {story.get('mood', '')} / {story.get('harmony', '')}")
            lines.append(f"Colour Engine        : {story.get('engine', '')}")
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
.block-container { padding-top:2rem; max-width:1380px; }
.hero { position:relative; overflow:hidden; background:linear-gradient(135deg,rgba(6,6,6,.98) 0%,rgba(13,13,13,.98) 54%,rgba(28,20,8,.96) 100%); border-radius:30px; padding:42px 44px; color:var(--text); border:1px solid var(--line-strong); box-shadow:0 34px 110px rgba(0,0,0,.62), inset 0 1px 0 rgba(255,255,255,.04); margin-bottom:24px; }
.hero:before { content:""; position:absolute; inset:0; background:linear-gradient(90deg,transparent 0%,rgba(242,211,122,.08) 50%,transparent 100%), repeating-linear-gradient(90deg,rgba(255,255,255,.018) 0,rgba(255,255,255,.018) 1px,transparent 1px,transparent 16px); pointer-events:none; }
.hero:after { content:""; position:absolute; right:-90px; bottom:-110px; width:360px; height:360px; border:1px solid rgba(214,170,69,.18); border-radius:50%; box-shadow:0 0 0 28px rgba(214,170,69,.025), 0 0 0 62px rgba(214,170,69,.018); pointer-events:none; }
.hero-inner { position:relative; z-index:1; }
.hero h1 { font-family:'Cinzel',serif; font-size:3.15rem; line-height:1.02; margin:0 0 14px 0; letter-spacing:-.035em; font-weight:900; background:linear-gradient(180deg,#fff8e5 0%,#d6aa45 48%,#8f6721 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { color:rgba(247,241,227,.76); font-size:1.06rem; margin:0; max-width:920px; }
.lux-mark { display:inline-flex; align-items:center; gap:10px; background:linear-gradient(135deg,rgba(214,170,69,.18),rgba(242,211,122,.06)); border:1px solid var(--line-strong); color:var(--gold-2); border-radius:999px; padding:8px 14px; font-weight:900; font-size:.78rem; margin-bottom:20px; letter-spacing:.16em; text-transform:uppercase; box-shadow:0 12px 28px rgba(0,0,0,.28); }
.card { background:linear-gradient(180deg,var(--panel) 0%,rgba(5,5,5,.97) 100%); border:1px solid var(--line); border-radius:26px; padding:25px; box-shadow:0 22px 64px rgba(0,0,0,.48), inset 0 1px 0 rgba(255,255,255,.035); color:var(--text); }
.metric-card { background:linear-gradient(180deg,rgba(214,170,69,.08) 0%,rgba(255,255,255,.025) 100%); border:1px solid rgba(214,170,69,.18); border-radius:18px; padding:15px 16px; min-height:90px; }
.metric-label { color:var(--dim); font-size:.74rem; text-transform:uppercase; letter-spacing:.11em; font-weight:900; }
.metric-value { color:var(--text); font-size:1.12rem; font-weight:900; margin-top:6px; }
.recipe-title { font-family:'Cinzel',serif; color:var(--gold-2); font-size:1.92rem; letter-spacing:-.025em; line-height:1.08; margin:12px 0 6px 0; font-weight:800; }
.badge { display:inline-block; padding:7px 11px; border-radius:999px; background:rgba(214,170,69,.10); color:var(--gold-2); font-size:.75rem; font-weight:900; border:1px solid rgba(214,170,69,.28); letter-spacing:.04em; }
.fit-badge { background:linear-gradient(135deg,var(--gold) 0%,var(--gold-dark) 100%); color:#080808; border-color:rgba(242,211,122,.52); }
.setting-row { display:flex; justify-content:space-between; align-items:center; gap:16px; border-bottom:1px solid rgba(214,170,69,.13); padding:12px 0; }
.setting-label { color:var(--muted); font-weight:650; font-size:.92rem; }
.setting-value { color:var(--text); font-weight:900; font-size:.98rem; text-align:right; }
.palette-dot { display:inline-block; width:34px; height:34px; border-radius:50%; border:2px solid rgba(214,170,69,.36); margin-right:7px; vertical-align:middle; box-shadow:0 9px 20px rgba(0,0,0,.42); }
.small-muted { color:var(--muted); font-size:.93rem; }
.variant-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:8px; }
.variant-card { border:1px solid rgba(214,170,69,.18); border-radius:18px; padding:14px; background:rgba(255,255,255,.025); }
.variant-card h4 { margin:0 0 8px 0; color:var(--gold-2); }
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
@media (max-width: 900px) { .hero h1{font-size:2.2rem;} .hero{padding:30px 26px;} .social-grid,.variant-grid{grid-template-columns:1fr 1fr;} }
@media (max-width: 560px) { .social-grid,.variant-grid{grid-template-columns:1fr;} }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="hero-inner">
        <div class="lux-mark">◆ FUJI LOOK ASSISTANT PRO</div>
        <h1>Black Gold Fujifilm<br/>JPEG direction studio.</h1>
        <p>Upload a reference image and receive a scene-aware, camera-aware Fujifilm recipe direction with a colour/tone/grain fingerprint engine and subtle, balanced, and strong variants for X-Trans I through X-Trans V.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🎛️ Camera Target")
    camera_model = st.selectbox("Camera model", list(CAMERA_MODEL_MAP.keys()), index=0)
    sensor_choice = st.radio("Fallback sensor generation", ["X-Trans V", "X-Trans IV", "X-Trans III", "X-Trans II", "X-Trans I"], index=0)
    fallback_code = sensor_choice.replace("X-Trans ", "")
    sensor_code, model_note = camera_to_sensor(camera_model, fallback_code)
    resolved_sensor = SENSOR_LABEL_MAP.get(sensor_code, sensor_choice)
    st.markdown(f"**Resolved target:** {resolved_sensor}")
    if model_note:
        st.caption(model_note)
    sims = SENSOR_SIM_MAP.get(sensor_code, XTRANS_V_SIMS)
    st.markdown(f"**{len(sims)}** film simulation menu items")
    st.markdown(f"**{len(sensor_presets(sensor_code))}** look directions")
    st.caption(sensor_capability_note(sensor_code))
    st.divider()
    st.markdown("## 🎯 Look Intention")
    look_intent = st.selectbox("Creative target", list(INTENT_PROFILES.keys()), index=0)
    st.caption("Use Auto for pure image analysis, or guide the app toward a specific creative direction.")
    st.divider()
    st.markdown("## 🧭 Studio Note")
    st.caption("This version uses scene-aware zone analysis, camera model mapping, recipe variants, a fingerprint layer, optional Colour Science, Color Story analysis, and a conservative tone/color guardrail engine to protect highlights, shadows, and WB shifts.")
    st.markdown(f"**Colour engine:** {'colour-science active' if HAS_COLOUR else 'OpenCV/NumPy fallback'}")

left, right = st.columns([.92, 1.08], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Upload reference image")
    uploaded = st.file_uploader("JPG, PNG, or WEBP", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        display_image = Image.open(uploaded)
        display_image = ImageOps.exif_transpose(display_image).convert("RGB")
        st.image(display_image, caption="Reference image", width="stretch")
        uploaded.seek(0)
        if st.button("Analyze Look", type="primary", width="stretch"):
            rgb = load_rgb(uploaded)
            features = extract_look_features(rgb)
            result = recommend(features, sensor_code, look_intent)
            st.session_state["result"] = result
            st.session_state["sensor_code"] = sensor_code
            st.session_state["camera_model"] = camera_model
            st.session_state["resolved_sensor"] = resolved_sensor
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
        st.markdown(f"<br><br><span class='badge'>Scene: {result.get('scene','general')} · {result.get('scene_confidence',0):.0%}</span>", unsafe_allow_html=True)
        st.markdown(f"&nbsp; <span class='badge'>Intent: {result.get('intent','Auto from image')}</span>", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        metrics = [
            ("Warmth", feats["warmth"]), ("Saturation", feats["sat"]),
            ("Contrast", feats["contrast"]), ("Softness", feats["softness"]),
            ("Skin Zone", feats.get("skin_ratio",0)), ("Sky / Blue", feats.get("sky_ratio",0)),
            ("Foliage", feats.get("foliage_ratio",0)), ("Dark Area", feats.get("dark_ratio",0)),
            ("Grain/Texture", feats.get("grain_signature",0)), ("Colour Split", abs(feats.get("highlight_warmth",.5)-feats.get("shadow_warmth",.5))*2),
            ("Analog Character", feats.get("analog_character",0)), ("Halation/Glow", feats.get("halation_score",0)),
            ("Matte Blacks", feats.get("matte_black_score",0)), ("Bloom/Diffusion", feats.get("bloom_score",0)),
        ]
        for i, (label, val) in enumerate(metrics):
            with (c1 if i % 2 == 0 else c2):
                st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{val:.0%}</div></div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        palette_html = "".join([f"<span class='palette-dot' style='background:{h}' title='{h}'></span>" for h in feats.get("palette", [])])
        st.markdown("**Dominant palette**")
        st.markdown(palette_html, unsafe_allow_html=True)
        story = feats.get("color_story", {}) or {}
        if story:
            st.markdown("### Color Story")
            st.markdown(f"<div class='barcode-strip' style='background:{story.get('barcode_css', 'linear-gradient(90deg,#111,#333)')}'></div>", unsafe_allow_html=True)
            st.markdown(f"<p class='small-muted'><strong>{story.get('mood','balanced colour')}</strong> · {story.get('harmony','balanced palette')} · Engine: {story.get('engine','fallback')}</p>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            story_metrics = [("Warm", story.get("warm_ratio",0)), ("Cool", story.get("cool_ratio",0)), ("Diversity", story.get("diversity",0)), ("Palette Contrast", story.get("palette_contrast",0)), ("Story Strength", story.get("story_score",0)), ("Neutral", story.get("neutral_ratio",0))]
            for idx, (label, val) in enumerate(story_metrics):
                with [s1, s2, s3][idx % 3]:
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{val:.0%}</div></div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    if "result" not in st.session_state:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What you’ll get")
        st.markdown("""
        - Camera model mapping for **X-Trans I, II, III, IV, and V**
        - Scene-aware analysis: portrait, landscape, cafe/product, night/neon, documentary, or B&W
        - Zone-aware interpretation: skin, sky/blue, foliage, highlights, shadows, and palette
        - NegClone-inspired fingerprint matching: tone curve, hue distribution, chroma, texture and split colour-cast
        - A practical recipe plus **Subtle / Balanced / Strong** variants
        - SpektraFilm-inspired film-process detector: halation, bloom, matte blacks, print warmth, dye muting, and grain texture
        - Optional Colour Science engine: sRGB → XYZ → CIE Lab and Delta E-based palette checks
        - KALMUS-inspired Color Story panel: palette barcode, colour mood, harmony, warm/cool balance, and colour diversity
        - An honest **in-camera achievability** label
        - TXT + JSON export
        """)
        st.warning("This is a look-direction assistant, not a guaranteed exact recipe reverse-engineer.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        result = st.session_state["result"]
        recipe = result["best"]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<span class='badge fit-badge'>{fit_label(result['best_score'])}</span>", unsafe_allow_html=True)
        st.markdown(f"<h2 class='recipe-title'>{recipe['name']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='small-muted'>{recipe.get('notes','')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='small-muted'><strong>Sensor compatibility:</strong> {recipe.get('sensor_note', '')}</p>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Film Sim</div><div class='metric-value'>{recipe['film_simulation']}</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Direction Fit</div><div class='metric-value'>{result['best_score']:.0%}</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Fingerprint Fit</div><div class='metric-value'>{result.get('fingerprint_fit',0):.0%}</div></div>", unsafe_allow_html=True)
        with m4:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Camera Target</div><div class='metric-value'>{st.session_state.get('resolved_sensor', resolved_sensor)}</div></div>", unsafe_allow_html=True)

        process = result.get("features", {}).get("film_process", {})
        if process:
            st.markdown("### Film process reading")
            flag_text = ", ".join(process.get("flags", [])) if process.get("flags") else "No strong analog-process artifacts detected"
            st.markdown(f"<p class='small-muted'>{flag_text}</p>", unsafe_allow_html=True)
            pc1, pc2, pc3 = st.columns(3)
            process_metrics = [("Analog Character", process.get("analog_character",0)), ("Halation / Glow", process.get("halation",0)), ("Matte Blacks", process.get("matte_black",0)), ("Print Warmth", process.get("print_warmth",0)), ("Dye Muting", process.get("dye_muting",0)), ("Film Grain", process.get("film_grain",0))]
            for idx, (label, val) in enumerate(process_metrics):
                with [pc1, pc2, pc3][idx % 3]:
                    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{val:.0%}</div></div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

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
                ("WB Shift", f"R {format_signed(recipe['wb_shift_r'])}, B {format_signed(recipe['wb_shift_b'])}"),
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

        st.markdown("### In-camera achievability")
        realism = result.get("realism", {})
        st.markdown(f"<span class='badge'>{realism.get('label','')}</span>", unsafe_allow_html=True)
        for w in realism.get("warnings", []):
            st.markdown(f"- {w}")

        st.markdown("### Look notes")
        for g in result.get("guidance", []):
            st.markdown(f"- {g}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recipe strength variants")
        st.markdown('<div class="variant-grid">', unsafe_allow_html=True)
        for var in result.get("variants", []):
            rows = compact_recipe_table(var)[:8]
            html = f"<div class='variant-card'><h4>{var.get('variant')}</h4>"
            for label, value in rows:
                html += f"<div class='setting-row'><span class='setting-label'>{label}</span><span class='setting-value'>{value}</span></div>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top alternatives")
        for alt in result.get("alternatives", [])[:4]:
            st.markdown(f"<div class='setting-row'><span class='setting-label'>{alt['name']}<br><small>{alt['best_for']}</small></span><span class='setting-value'>{alt['film_simulation']}<br>{alt['fit']:.0%} · FP {alt.get('fingerprint_fit',0):.0%}</span></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Export")
        export_json = {
            "app": "Fuji Look Assistant Pro",
            "version": APP_VERSION,
            "camera_model": st.session_state.get("camera_model", camera_model),
            "sensor": st.session_state.get("resolved_sensor", resolved_sensor),
            "intent": result.get("intent"),
            "scene": result.get("scene"),
            "recipe": recipe,
            "variants": result.get("variants", []),
            "fingerprint_fit": result.get("fingerprint_fit", 0.0),
            "color_story": result.get("features", {}).get("color_story", {}),
            "match_details": result.get("match_details", {}),
            "visual_summary": result["visual_summary"],
            "realism": result.get("realism", {}),
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
