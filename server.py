# backend/main.py
import io
import json
import asyncio
from typing import List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")  # or a custom weights path
DEVICE = os.environ.get("DEVICE", "cpu")  # "cpu" or "cuda"
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.3))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 2))

# Load model (blocking on startup)
print(f"Loading model {MODEL_PATH} on device {DEVICE} ...")
model = YOLO(MODEL_PATH)
# ultralytics usually auto-selects device; you can force using .to() if desired:
try:
    if DEVICE.lower().startswith("cuda"):
        model.to("cuda")
except Exception as e:
    print("Could not move model to CUDA:", e)
print("Model loaded.")

# Use a small threadpool to run blocking inference so ASGI loop isn't blocked.
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def pil_bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes -> RGB numpy array (H,W,3)"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def run_inference_on_image(np_img: np.ndarray) -> List[Dict]:
    """
    Run the YOLO model on an RGB numpy array and return detections in normalized coords:
    [{label, confidence, x, y, w, h}, ...] where x,y are top-left corner normalized (0..1).
    """
    # ultralytics accepts numpy arrays directly
    results = model(np_img, verbose=False, conf=CONF_THRESHOLD)
    # results is a list-like; take first
    if len(results) == 0:
        return []

    res = results[0]
    # res.boxes contains coordinates in xyxy format (pixels)
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    # image shape
    h, w = np_img.shape[:2]
    detections = []

    # boxes.xyxy, boxes.conf, boxes.cls
    xyxy = boxes.xyxy.cpu().numpy()  # shape (N,4)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.ones(len(xyxy))
    cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else np.zeros(len(xyxy))
    names = res.names if hasattr(res, "names") else {}

    for (x1, y1, x2, y2), conf, c in zip(xyxy, confs, cls):
        bw = x2 - x1
        bh = y2 - y1
        # normalize
        nx = float(x1 / w)
        ny = float(y1 / h)
        nw = float(bw / w)
        nh = float(bh / h)
        label = names.get(int(c), str(int(c)))
        detections.append({
            "label": str(label),
            "confidence": float(conf),
            "x": nx,
            "y": ny,
            "w": nw,
            "h": nh
        })
    return detections


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """
    Expects binary messages which are JPEG frames.
    Sends back JSON messages like: { "detections": [ {label, confidence, x, y, w, h}, ... ] }
    """
    await websocket.accept()
    print("Client connected.")
    try:
        while True:
            msg = await websocket.receive()
            # WebSocket frame can be: {"type":"websocket.receive", "bytes": ... } or {"text":...}
            image_bytes = None
            if "bytes" in msg:
                image_bytes = msg["bytes"]
            elif "text" in msg:
                # optionally backend can accept base64 data URLs or JSON; try to parse JSON with field 'image' (base64)
                text = msg["text"]
                try:
                    payload = json.loads(text)
                    if isinstance(payload, dict) and "image" in payload:
                        # image expected as base64 string like "data:image/jpeg;base64,...." or plain base64
                        import base64
                        b = payload["image"]
                        if b.startswith("data:"):
                            b = b.split(",", 1)[1]
                        image_bytes = base64.b64decode(b)
                except Exception:
                    image_bytes = None

            if not image_bytes:
                # nothing to do; continue listening
                await asyncio.sleep(0.001)
                continue

            # Decode image in threadpool if needed
            loop = asyncio.get_event_loop()
            try:
                np_img = await loop.run_in_executor(executor, pil_bytes_to_numpy, image_bytes)
            except Exception as e:
                print("Failed to decode image:", e)
                # respond with empty detections
                await websocket.send_text(json.dumps({"detections": []}))
                continue

            # Run inference in threadpool (blocking)
            try:
                detections = await loop.run_in_executor(executor, run_inference_on_image, np_img)
            except Exception as ex:
                print("Inference error:", ex)
                detections = []

            # Send result back
            payload = {"detections": detections}
            try:
                await websocket.send_text(json.dumps(payload))
            except Exception as e:
                print("Failed to send result:", e)
                break

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        print("Cleaning up connection.")


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
