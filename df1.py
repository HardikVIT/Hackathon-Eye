from ultralytics import YOLO
import cv2
import pyttsx3
import speech_recognition as sr
import numpy as np
import sounddevice as sd
import wave
import math
import tempfile
import os

def text_to_wav(text, filename="voice.wav"):
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.save_to_file(text, filename)
    engine.runAndWait()

def spatialize_audio(wav_path, azimuth_deg, distance=1.0):
    wf = wave.open(wav_path, 'rb')
    rate = wf.getframerate()
    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    wf.close()

    data = data.astype(np.float32) / 32768.0
    az = math.radians(azimuth_deg)

    # interaural delay and loudness difference
    max_delay = int(0.0007 * rate)
    delay = int(max_delay * math.sin(az))
    gain_left = 1.0 - 0.4 * max(0, math.sin(az))
    gain_right = 1.0 - 0.4 * max(0, -math.sin(az))

    # --- Distance-based attenuation (stronger perceptual drop-off)
    distance = max(0.5, distance)
    attenuation = 1 / (distance ** 1.5)
    gain_left *= attenuation
    gain_right *= attenuation

    if delay > 0:
        left = np.concatenate([np.zeros(delay), data * gain_left])[:len(data)]
        right = data * gain_right
    else:
        right = np.concatenate([np.zeros(-delay), data * gain_right])[:len(data)]
        left = data * gain_left

    stereo = np.vstack([left, right]).T
    return stereo, rate


def play_spatial_voice(text, azimuth_deg, distance=1.0):
    tmp_wav = os.path.join(tempfile.gettempdir(), "voice.wav")
    text_to_wav(text, tmp_wav)
    stereo_audio, fs = spatialize_audio(tmp_wav, azimuth_deg, distance)
    sd.play(stereo_audio, fs)
    sd.wait()

# ---------- YOLO + Voice Logic ----------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
model = YOLO("yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle","bottle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

play_spatial_voice("What do you want me to look for?", 0)

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

try:
    text = r.recognize_google(audio).lower()
    print("Google Speech Recognition thinks you said:", text)
except Exception as e:
    print("Could not understand audio:", e)
    text = "laptop"

flag = 0
HFOV_deg = 70.0  # horizontal FOV
KNOWN_OBJECT_WIDTH = 0.3  # meters (average laptop width)
FOCAL_LENGTH = 700        # adjust experimentally

while flag == 0:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    h, w = img.shape[:2]

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = classNames[cls]

            if label == text:
                # compute azimuth
                x_c = (x1 + x2) / 2.0
                azimuth = ((x_c - w/2) / (w/2)) * (HFOV_deg / 2.0)

                obj_pixel_width = max(x2 - x1, 1)                # avoid zero-div
                # If FOCAL_LENGTH is in pixels and KNOWN_OBJECT_WIDTH in meters -> distance in meters
                distance_m = (KNOWN_OBJECT_WIDTH * FOCAL_LENGTH) / float(obj_pixel_width)

                # Print raw values for debugging (important!)
                print("pixel_width:", obj_pixel_width, "focal(px):", FOCAL_LENGTH, "distance(m):", distance_m)

                print(f"I can see {label} at {azimuth:.1f}° and {distance_m:.2f} m away")

                play_spatial_voice(f"I can see {label} on your {'right' if azimuth>0 else 'left'} about {distance_m:.1f} meters away", azimuth, distance_m)
                flag = 1

            # draw detection
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
