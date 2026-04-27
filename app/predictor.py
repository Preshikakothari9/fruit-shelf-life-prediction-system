"""
FruitFresh Predictor — colour-analysis CNN surrogate.

Analyses the dominant hue/saturation of an uploaded image and maps it to a
known fruit type.  A comprehensive dictionary then supplies shelf-life
estimates, confidence scores, freshness status, and storage tips.

When a real Keras .h5 model is available it can be hot-swapped; until then
the colour-based heuristic provides plausible results for demos.
"""

import json
import os

import numpy as np
from PIL import Image

# ── Fruit knowledge base ────────────────────────────────────────────────
FRUIT_DB = {
    "Apple": {
        "days": (4, 9), "emoji": "🍎", "temp": "1-4 °C",
        "tip": "Store in the crisper drawer. Keep away from other produce — apples emit ethylene.",
        "hues": [(0, 25), (340, 360)], "sat": 0.30,
    },
    "Banana": {
        "days": (2, 7), "emoji": "🍌", "temp": "15-20 °C",
        "tip": "Room temperature until ripe, then refrigerate. Skin darkens but flesh stays fresh.",
        "hues": [(40, 70)], "sat": 0.35,
    },
    "Orange": {
        "days": (5, 9), "emoji": "🍊", "temp": "3-4 °C",
        "tip": "Refrigerate in a mesh bag for air circulation. Lasts ~1 week at room temperature.",
        "hues": [(15, 40)], "sat": 0.50,
    },
    "Strawberry": {
        "days": (3, 7), "emoji": "🍓", "temp": "0-2 °C",
        "tip": "Store unwashed on paper towels in the fridge. Remove stems only when ready to eat.",
        "hues": [(345, 360), (0, 15)], "sat": 0.50,
    },
    "Grape": {
        "days": (5, 9), "emoji": "🍇", "temp": "0-2 °C",
        "tip": "Keep unwashed in a ventilated bag in the fridge. Wash just before eating.",
        "hues": [(260, 320)], "sat": 0.15,
    },
    "Mango": {
        "days": (4, 9), "emoji": "🥭", "temp": "10-13 °C",
        "tip": "Ripen at room temp; once soft, refrigerate.",
        "hues": [(25, 55)], "sat": 0.45,
    },
    "Watermelon": {
        "days": (5, 9), "emoji": "🍉", "temp": "4 °C",
        "tip": "Whole: room temp. Cut: wrap tightly and refrigerate.",
        "hues": [(100, 150)], "sat": 0.25,
    },
    "Blueberry": {
        "days": (4, 9), "emoji": "🫐", "temp": "0-1 °C",
        "tip": "Refrigerate dry and unwashed. Discard any mouldy berries.",
        "hues": [(220, 270)], "sat": 0.20,
    },
    "Lemon": {
        "days": (5, 9), "emoji": "🍋", "temp": "4 °C",
        "tip": "Sealed bag in the fridge.",
        "hues": [(50, 70)], "sat": 0.55,
    },
    "Kiwi": {
        "days": (5, 9), "emoji": "🥝", "temp": "0-2 °C",
        "tip": "Unripe: room temp. Ripe: refrigerate. Keep away from ethylene.",
        "hues": [(70, 110)], "sat": 0.30,
    },
    "Peach": {
        "days": (3, 5), "emoji": "🍑", "temp": "0-1 °C",
        "tip": "Ripen at room temp stem-side down. Once ripe, refrigerate.",
        "hues": [(10, 35)], "sat": 0.35,
    },
    "Avocado": {
        "days": (3, 7), "emoji": "🥑", "temp": "5-7 °C",
        "tip": "Ripen at room temp. Refrigerate ripe avocados.",
        "hues": [(60, 130)], "sat": 0.15,
    },
    "Cherry": {
        "days": (4, 9), "emoji": "🍒", "temp": "0-2 °C",
        "tip": "Refrigerate unwashed in high humidity.",
        "hues": [(340, 360), (0, 10)], "sat": 0.40,
    },
    "Pineapple": {
        "days": (3, 5), "emoji": "🍍", "temp": "7-10 °C",
        "tip": "Store ripe pineapple upside-down in the fridge.",
        "hues": [(40, 65)], "sat": 0.45,
    },
    "Pomegranate": {
        "days": (5, 9), "emoji": "🔴", "temp": "5 °C",
        "tip": "Whole fruit at 5 °C. Seeds in an airtight container.",
        "hues": [(340, 360), (0, 15)], "sat": 0.35,
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _dominant_hue_sat(image_path: str, sample_size: int = 2000):
    """Return (dominant_hue 0-360, mean_saturation 0-1, mean_brightness 0-1)."""
    img = Image.open(image_path).convert("RGB").resize((150, 150))
    pixels = np.array(img).reshape(-1, 3).astype(float)

    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[idx]

    # RGB → HSV (manual, avoids opencv dependency)
    r, g, b = pixels[:, 0] / 255, pixels[:, 1] / 255, pixels[:, 2] / 255
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    hue = np.zeros_like(delta)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Saturation
    sat = np.where(cmax > 0, delta / cmax, 0)

    # Filter out very dark / very light pixels (background)
    bright = cmax
    keep = (bright > 0.15) & (bright < 0.95) & (sat > 0.08)
    if keep.sum() < 50:
        keep = np.ones(len(hue), dtype=bool)

    return float(np.median(hue[keep])), float(np.mean(sat[keep])), float(np.mean(bright[keep]))


def _hue_distance(h1: float, h2: float) -> float:
    """Circular distance between two hues in degrees (0-180)."""
    d = abs(h1 - h2) % 360
    return min(d, 360 - d)


def _match_fruit(hue: float, sat: float):
    """Find the best-matching fruit using circular hue distance + saturation fit.

    Returns (fruit_name, score 0-1). Higher score = better match.
    """
    best_name = None
    best_score = -1.0

    for name, info in FRUIT_DB.items():
        # Closest distance to any of this fruit's hue bands
        band_dist = min(
            _hue_distance(hue, (lo + hi) / 2 if hi >= lo else (lo + hi + 360) / 2 % 360)
            for lo, hi in info["hues"]
        )
        # Hue similarity: 1.0 when on centre, 0 when >=60° away
        hue_score = max(0.0, 1.0 - band_dist / 60.0)

        # Saturation fit: penalise if image is much less saturated than expected
        sat_score = min(1.0, sat / max(info["sat"], 1e-3))

        score = 0.7 * hue_score + 0.3 * sat_score
        if score > best_score:
            best_name, best_score = name, score

    return best_name, best_score


def _freshness_status(days: float):
    if days >= 8:
        return "Fresh"
    if days >= 5:
        return "Good"
    if days >= 3:
        return "Use Soon"
    if days > 0:
        return "Use Immediately"
    return "Rotten"


# ── Public API ───────────────────────────────────────────────────────────

def _normalise_class_name(raw: str):
    """Map a raw class label like 'rotten_banana' / 'freshApples' to (fruit, is_rotten)."""
    s = raw.strip().lower().replace("-", "_").replace(" ", "_")
    is_rotten = "rotten" in s
    s = s.replace("fresh_", "").replace("rotten_", "")
    s = s.replace("fresh", "").replace("rotten", "")
    s = s.strip("_")
    # Strip trailing plural 's' if needed (apples -> apple)
    if s.endswith("s") and s[:-1].capitalize() in FRUIT_DB:
        s = s[:-1]
    fruit = s.capitalize()
    if fruit not in FRUIT_DB:
        fruit = None
    return fruit, is_rotten


# Cache the Keras model so we don't reload it on every request.
_KERAS_MODEL = None
_KERAS_CLASSES = None


def _load_keras_model():
    global _KERAS_MODEL, _KERAS_CLASSES
    if _KERAS_MODEL is not None:
        return _KERAS_MODEL, _KERAS_CLASSES

    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "ml_model", "fruit_model.h5")
    classes_path = os.path.join(base, "ml_model", "classes.json")

    if not (os.path.exists(model_path) and os.path.exists(classes_path)):
        return None, None

    # Skip the placeholder JSON file the bootstrap creates.
    try:
        with open(model_path, "rb") as f:
            head = f.read(8)
        if not head.startswith(b"\x89HDF") and not head.startswith(b"PK"):
            return None, None
    except OSError:
        return None, None

    try:
        from tensorflow.keras.models import load_model
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)["classes"]
        _KERAS_MODEL = load_model(model_path)
        _KERAS_CLASSES = classes
        return _KERAS_MODEL, _KERAS_CLASSES
    except Exception as e:
        print("[predictor] Keras model unavailable, using heuristic:", e)
        return None, None


def _shelf_life_from_freshness(info: dict, freshness: float) -> float:
    """Map a 0-1 freshness factor to days within the fruit's shelf-life band."""
    lo, hi = info["days"]
    f = max(0.0, min(1.0, freshness))
    return round(lo + (hi - lo) * f, 1)


def predict_freshness(image_path: str) -> dict:
    """Analyse a fruit image and return shelf-life prediction (deterministic)."""
    # Always compute colour features — they drive freshness even when the CNN runs.
    hue, sat, bright = _dominant_hue_sat(image_path)

    # Rotten heuristic from pixels: very dark or muddy/desaturated browns.
    is_rotten_pixels = (bright < 0.28) or (sat < 0.18 and bright < 0.45)

    fruit = None
    confidence = None

    model, classes = _load_keras_model()
    if model is not None:
        try:
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            img = load_img(image_path, target_size=(224, 224))
            arr = img_to_array(img)
            arr = preprocess_input(arr)
            arr = np.expand_dims(arr, axis=0)
            preds = model.predict(arr, verbose=0)[0]
            best_idx = int(np.argmax(preds))
            confidence = float(preds[best_idx])
            predicted_class = classes[best_idx]
            mapped_fruit, is_rotten_label = _normalise_class_name(predicted_class)
            if mapped_fruit is not None:
                fruit = mapped_fruit
                # Trust the trained model's rotten label over pixel heuristic.
                is_rotten_pixels = is_rotten_label
        except Exception as e:
            print("[predictor] Inference failed, using heuristic:", e)

    if fruit is None:
        fruit, match_score = _match_fruit(hue, sat)
        # Convert match score (0-1) into a confidence estimate.
        confidence = round(0.55 + 0.40 * match_score, 3)

    info = FRUIT_DB[fruit]

    if is_rotten_pixels:
        predicted_days = 0.0
        status = "Rotten"
    else:
        # Brighter & more saturated -> closer to the upper end of the shelf-life band.
        freshness_factor = 0.5 * min(1.0, sat / max(info["sat"], 1e-3)) + 0.5 * max(0.0, min(1.0, (bright - 0.3) / 0.6))
        predicted_days = _shelf_life_from_freshness(info, freshness_factor)
        status = _freshness_status(predicted_days)

    return {
        "fruit_type": fruit,
        "predicted_days": predicted_days,
        "confidence": round(float(confidence), 3),
        "freshness_status": status,
        "temperature": info.get("temp", "Room Temp"),
        "storage_tip": info["tip"],
        "emoji": info["emoji"],
    }


def create_dummy_model(path: str):
    """Write a small placeholder file so the app boots without errors."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "type": "FruitFresh-dummy-v1",
        "classes": list(FRUIT_DB.keys()),
        "note": "Replace with a real Keras .h5 for production use.",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
