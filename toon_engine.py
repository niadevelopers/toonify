"""
Toonify AI engine: photo -> anime/Naruto-style character, 100% local. Needs an NVIDIA GPU (8GB+ VRAM).

Setup (once):
    pip install torch diffusers transformers accelerate safetensors fastapi uvicorn opencv-python-headless numpy pillow
Run:
    uvicorn toon_engine:app --host 127.0.0.1 --port 8000
Then expose it (free):  cloudflared tunnel --url http://localhost:8000
and put the https://....trycloudflare.com address in Render as TOON_ENGINE_URL.

How it keeps your likeness: ControlNet locks your photo's outlines/pose, IP-Adapter feeds your face back in,
a quick local check scores layout + colors and retries tighter if needed, and colors are nudged back to the original.
Optional env: BASE_MODEL (any SD1.5 anime checkpoint), LONG_SIDE (768), LOW_VRAM=1 (6GB cards), USE_IP_ADAPTER=0
"""
import io
import os
import threading
import zlib

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, ImageOps

from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

BASE_MODEL = os.getenv("BASE_MODEL", "lambdalabs/sd-naruto-diffusers")
LONG_SIDE = int(os.getenv("LONG_SIDE", "768"))
USE_IP = os.getenv("USE_IP_ADAPTER", "1") == "1"
LOW_VRAM = os.getenv("LOW_VRAM", "0") == "1"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device == "cuda" else torch.float32
STEPS = 28 if device != "cpu" else 14
if device == "cpu":
    print("[warn] No GPU found: expect minutes per image.")

PROMPT = ("anime character illustration, Naruto shonen manga style, cel shading, bold clean black outlines, "
          "flat vibrant colors, expressive eyes, detailed face, same pose, same clothing, same background")
NEGATIVE = "photo, photorealistic, realistic skin texture, 3d render, blurry, lowres, deformed face, extra fingers, text, watermark"
SCHEDULE = [(0.60, 1.0), (0.45, 1.3)]  # (img2img strength, outline-lock strength): retry holds the photo tighter

print("[toon] loading models (first run downloads a few GB) ...")
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=dtype)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(BASE_MODEL, controlnet=controlnet, torch_dtype=dtype)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
if USE_IP:
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
    pipe.set_ip_adapter_scale(0.6)
if LOW_VRAM and device == "cuda":
    pipe.enable_model_cpu_offload()
else:
    pipe.to(device)
pipe.set_progress_bar_config(disable=True)
lock = threading.Lock()


def canny(rgb):
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    m = float(np.median(g))
    e = cv2.Canny(g, int(max(20, 0.66 * m)), int(min(250, 1.33 * m + 40)))
    return Image.fromarray(np.stack([e] * 3, -1))


def pearson(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def score(orig, out):
    def edges(x):
        e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (5, 5), 0), 60, 140)
        return cv2.resize(cv2.GaussianBlur(e, (0, 0), 6), (64, 64), interpolation=cv2.INTER_AREA)

    def lab(x):
        return cv2.cvtColor(cv2.resize(x, (24, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).astype(np.float32)
    structure = max(0.0, pearson(edges(orig), edges(out)))
    color = float(np.clip(1 - np.abs(lab(orig) - lab(out)).mean() / 55, 0, 1))
    return 0.5 * structure + 0.5 * color


def color_pull(out, orig, alpha=0.35):
    o = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor(cv2.resize(orig, (out.shape[1], out.shape[0])), cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        matched = (o[..., c] - o[..., c].mean()) * ((r[..., c].std() + 1e-6) / (o[..., c].std() + 1e-6)) + r[..., c].mean()
        o[..., c] = (1 - alpha) * o[..., c] + alpha * matched
    return cv2.cvtColor(np.clip(o, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def make_toon(data: bytes) -> bytes:
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image")
    s = LONG_SIDE / max(pil.size)
    pil = pil.resize((max(64, round(pil.width * s / 8) * 8), max(64, round(pil.height * s / 8) * 8)), Image.LANCZOS)
    orig, control, seed = np.array(pil), canny(np.array(pil)), zlib.crc32(data) & 0x7FFFFFFF

    best = None
    with lock:
        for i, (strength, cscale) in enumerate(SCHEDULE):
            kw = dict(prompt=PROMPT, negative_prompt=NEGATIVE, image=pil, control_image=control, strength=strength,
                      controlnet_conditioning_scale=cscale, num_inference_steps=STEPS, guidance_scale=7.0,
                      generator=torch.Generator(device="cpu").manual_seed(seed + i))
            if USE_IP:
                kw["ip_adapter_image"] = pil
            out = color_pull(np.array(pipe(**kw).images[0]), orig)
            sc = score(orig, out)
            print(f"[toon] attempt {i + 1}: fidelity {sc:.2f}")
            if best is None or sc > best[0]:
                best = (sc, out)
            if sc >= 0.72:
                break
    buf = io.BytesIO()
    Image.fromarray(best[1]).save(buf, format="PNG")
    return buf.getvalue()


app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "device": device}


@app.post("/toon")
async def toon(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(400, "Send the image as the request body")
    return Response(await run_in_threadpool(make_toon, data), media_type="image/png")
