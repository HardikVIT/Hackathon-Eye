from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
import json
import torch
import uvicorn
from ultralytics import YOLO
import pyttsx3
from pydub import AudioSegment

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# -----------------------------
# YOLOv8 model setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")
model = YOLO("yolov8n.pt").to(device)

# -----------------------------
# TTS engine
# -----------------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# -----------------------------
# WebSocket endpoint
# -----------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("📡 Client connected")
    target_objects = []

    try:
        # First message should be target objects
        first_msg = await ws.receive_text()
        data = json.loads(first_msg)
        target_objects = [obj.lower() for obj in data.get("target_objects", [])]
        print(f"🎯 Detecting only: {target_objects}")

        while True:
            # Receive frame from frontend
            frame_bytes = await ws.receive_bytes()
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # YOLO inference
            results = model(frame, verbose=False)
            detections = []

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id].lower()

                if label in target_objects:
                    detections.append({
                        "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                        "confidence": conf, "name": label
                    })

            # Generate spatial audio
            audio_data, audio_texts = generate_spatial_audio(detections, frame.shape[1], frame.shape[0])

            # Convert to WAV bytes
            buffer = io.BytesIO()
            audio_data.export(buffer, format="wav")
            buffer.seek(0)
            await ws.send_bytes(buffer.read())

            # Terminal log
            if audio_texts:
                print(f"🔊 Audio sent for: {', '.join(audio_texts)}")
            else:
                print("🔊 No target objects detected; silent audio sent.")

    except Exception as e:
        print("❌ Connection closed:", e)
    finally:
        await ws.close()
        print("🔌 WebSocket closed")

# -----------------------------
# Generate TTS-based spatial audio
# -----------------------------
def generate_spatial_audio(detections, frame_width, frame_height):
    """
    Create TTS-based spatial audio for detected objects.
    Returns AudioSegment and list of texts.
    """
    if not detections:
        # Return 0.3s silent audio
        return AudioSegment.silent(duration=300), []

    combined_audio = AudioSegment.silent(duration=0)
    audio_texts = []

    for det in detections:
        x_center = (det["xmin"] + det["xmax"]) / 2
        width = det["xmax"] - det["xmin"]
        height = det["ymax"] - det["ymin"]

        # Azimuth: -1 (left) → +1 (right)
        pan = (x_center / frame_width) * 2 - 1

        # Distance (rough, smaller box → farther)
        box_area = width * height
        area_norm = np.clip(box_area / (frame_width * frame_height), 0.01, 1)
        distance_text = f"{(1-area_norm)*10:.1f} meters"

        text = f"{det['name']} detected at {int((pan+1)*90)} degrees {'right' if pan>0 else 'left'} at {distance_text}"
        audio_texts.append(text)
        print("TTS:", text)

        # Generate TTS segment
        tts_engine.save_to_file(text, "temp.wav")
        tts_engine.runAndWait()
        segment = AudioSegment.from_wav("temp.wav")

        # Pan audio
        segment = segment.pan(pan)

        # Adjust volume by distance
        segment = segment - (10*(1-area_norm))

        combined_audio += segment

    return combined_audio, audio_texts

# -----------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
