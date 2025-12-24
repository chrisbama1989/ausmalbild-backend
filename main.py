from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True}

def fit_max(img, max_size: int):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_size:
        return img
    s = max_size / float(m)
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def adjust_gray(gray, brightness: int, contrast: float):
    g = gray.astype(np.float32)
    g = (g - 128.0) * float(contrast) + 128.0 + float(brightness)
    return np.clip(g, 0, 255).astype(np.uint8)

def xdog(gray, sigma: float, k: float, gamma: float, eps: float = 0.02, phi: float = 12.0):
    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma * k, sigmaY=sigma * k)

    d = g1.astype(np.float32) - gamma * g2.astype(np.float32)
    d = d / 255.0

    h = np.ones_like(d, dtype=np.float32)
    mask = d >= eps
    h[mask] = 1.0
    h[~mask] = 1.0 + np.tanh(phi * (d[~mask] - eps))

    out = (255.0 * (1.0 - h)).clip(0, 255).astype(np.uint8)
    return out

def to_binary(line_gray: np.ndarray, strength: int):
    s = int(max(0, min(80, strength)))
    block = 21 + 2 * (s // 8)
    if block % 2 == 0:
        block += 1
    C = 6 + (s // 10)
    return cv2.adaptiveThreshold(
        line_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, C
    )

def thicken_black(binary_white_bg: np.ndarray, passes: int):
    p = int(max(0, min(3, passes)))
    if p <= 0:
        return binary_white_bg
    inv = 255 - binary_white_bg
    k = np.ones((3, 3), np.uint8)
    inv = cv2.dilate(inv, k, iterations=p)
    return 255 - inv

def cleanup(binary_white_bg: np.ndarray, amount: int):
    a = int(max(0, min(80, amount)))
    if a <= 0:
        return binary_white_bg
    it = 1 if a <= 25 else (2 if a <= 55 else 3)
    inv = 255 - binary_white_bg
    k = np.ones((3, 3), np.uint8)
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k, iterations=it)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, k, iterations=max(1, it - 1))
    return 255 - inv

def preset_params(name: str):
    n = (name or "foto").lower().strip()

    presets = {
        "foto": {
            "maxSize": 1200,
            "brightness": 8,
            "contrast": 1.25,
            "sigma": 1.2,
            "k": 1.6,
            "gamma": 0.98,
            "strength": 40,
            "thicken": 1,
            "speckle": 22,
        },
        "wenig": {
            "maxSize": 1200,
            "brightness": 10,
            "contrast": 1.30,
            "sigma": 1.6,
            "k": 1.9,
            "gamma": 1.05,
            "strength": 55,
            "thicken": 1,
            "speckle": 35,
        },
        "zeichnung": {
            "maxSize": 1400,
            "brightness": 0,
            "contrast": 1.10,
            "sigma": 0.9,
            "k": 1.4,
            "gamma": 0.92,
            "strength": 25,
            "thicken": 1,
            "speckle": 10,
        },
    }

    return presets.get(n, presets["foto"])

@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    preset: str = Query("foto"),
):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return Response(status_code=400)

    p = preset_params(preset)

    img = fit_max(img, int(p["maxSize"]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = adjust_gray(gray, int(p["brightness"]), float(p["contrast"]))

    line_gray = xdog(
        gray,
        sigma=float(p["sigma"]),
        k=float(p["k"]),
        gamma=float(p["gamma"]),
        eps=0.02,
        phi=12.0
    )

    bw = to_binary(line_gray, int(p["strength"]))
    bw = thicken_black(bw, int(p["thicken"]))
    bw = cleanup(bw, int(p["speckle"]))

    ok, png = cv2.imencode(".png", bw)
    if not ok:
        return Response(status_code=500)

    return Response(content=png.tobytes(), media_type="image/png")
