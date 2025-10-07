from ultralytics import YOLO
import cv2
import pyttsx3
import speech_recognition as sr
import threading

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# COCO classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 160)

# Shared state
current_objects = set()
previous_objects = set()

# Thread-safe TTS speaker
def speak(text):
    engine.say(text)
    engine.runAndWait()

speak("What do you want me to look for?")

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something!")
    r.adjust_for_ambient_noise(source)  # Adjust for background noise
    audio = r.listen(source)

text = r.recognize_google(audio)
print("Google Speech Recognition thinks you said: " + text)
flag=0
while (flag==0):
    success, img = cap.read()
    results = model(img, stream=True)

    # registry of what’s in the current frame
    current_objects.clear()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = classNames[cls]
            if(label==text):
                print("I can see "+text+" now turning off the application")
                flag=1
                speak("I can see laptop now turning off the application")
            # add to registry
            current_objects.add(label)

            # draw
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    # Speaks for all objects
    # if current_objects != previous_objects:
    #     if current_objects:
    #         msg = "I can see " + ", ".join(current_objects)
    #     else:
    #         msg = "I see nothing now"
    #     print("Speaking:", msg)
    #     speak(msg)
    #     previous_objects = current_objects.copy()

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
