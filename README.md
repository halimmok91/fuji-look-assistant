# Fuji Look Assistant

**Fuji Look Assistant** is a professional Streamlit web app that analyzes a reference image and suggests a Fujifilm-style JPEG recipe direction for Fuji X-Series cameras.

It is designed as a creative look assistant, not an exact film recipe reverse-engineering tool. The app studies the look, feel, colour, tone, mood, and visual character of a reference image, then recommends a practical Fujifilm recipe starting point.

---

## Features

- Upload a reference image
- Analyze overall colour, tone, contrast, warmth, saturation, and mood
- Camera model selector
- Supports X-Trans I, II, III, IV, and V generations
- Sensor-safe recipe generation
- Subtle / Balanced / Strong recipe variants
- Scene-aware image analysis
- Zone-aware analysis for skin, sky, foliage, highlights, and shadows
- Fuji Look Fingerprint Engine
- Film Process Detector
- Optional Colour Science Engine using `colour-science`
- KALMUS-inspired Color Story Panel
- Dominant colour barcode / palette strip
- In-camera achievability warning
- TXT recipe export
- JSON analysis export
- Premium black and gold professional UI
- Social media footer section

---

## What the App Does

Fuji Look Assistant helps answer:

> “What Fujifilm recipe direction should I use to get close to this reference image?”

The app analyzes the uploaded image and recommends a Fujifilm look direction based on:

- brightness
- contrast
- dynamic range
- saturation
- warmth / coolness
- green / magenta tint
- shadow and highlight colour cast
- palette mood
- film-process character
- analog-style cues
- scene type
- selected Fujifilm camera model

It then generates a practical JPEG recipe for the selected camera generation.

---

## Important Disclaimer

Fuji Look Assistant does **not** guarantee an exact match to a reference image.

Final Fujifilm JPEG output depends on:

- camera model
- sensor generation
- firmware
- lens
- lighting
- exposure
- white balance
- ISO
- dynamic range
- subject colours
- scene conditions
- whether the reference image was edited

The app is intended to provide a strong creative starting point and recipe direction.

---

## Supported Camera Generations

The app supports:

- X-Trans I
- X-Trans II
- X-Trans III
- X-Trans IV
- X-Trans V

It also includes camera-model-aware compatibility logic, so older cameras do not receive settings they do not support.

---

## Example Camera Support

Examples of supported Fujifilm camera families include:

- X-Pro1
- X-E1
- X-M1
- X-T1
- X-T10
- X-E2 / X-E2S
- X100S / X100T
- X-Pro2
- X-T2
- X-T20
- X-E3
- X-H1
- X100F
- X-T3
- X-T30
- X-Pro3
- X100V
- X-T4
- X-S10
- X-E4
- X-T30 II
- X-H2S
- X-H2
- X-T5
- X-S20
- X100VI
- X-T50
- X-M5
- X-E5
- X-T30 III

Availability of some film simulations and JPEG settings may vary depending on firmware and exact body model.

---

## Main App File

For the latest version, use:

```txt
fuji_look_assistant_black_gold_colour_story_v4.py
