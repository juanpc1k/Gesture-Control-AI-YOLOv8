import cv2
import time
import random
import mediapipe as mp
from pathlib import Path

# Configuración de MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# -------------------------
# CONFIGURACIÓN
# -------------------------
DATASET_PATH = Path("dataset_hands")
CLASSES = ["palm", "fist", "thumbs_up"]
MAX_IMAGES_PER_CLASS = 500
SPLIT_RATIO = 0.8 
CAPTURE_DELAY = 0.2

# CREAR ESTRUCTURA LIMPIA (Solo lo que YOLO necesita)
for split in ["train", "val"]:
    (DATASET_PATH / split / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_PATH / split / "labels").mkdir(parents=True, exist_ok=True)

# -------------------------
# CARGAR MODELO
# -------------------------
base_options = BaseOptions(model_asset_path="hand_landmarker.task")
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = HandLandmarker.create_from_options(options)

contador = {clase: 0 for clase in CLASSES}
clase_actual = 0
last_capture = 0

def save_yolo_data(frame, hand_landmarks, class_id):
    split = "train" if random.random() < SPLIT_RATIO else "val"
    
    img_path = DATASET_PATH / split / "images"
    lbl_path = DATASET_PATH / split / "labels"

    # Coordenadas YOLO
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    x_min, y_min, x_max, y_max = max(0, min(xs)), max(0, min(ys)), min(1, max(xs)), min(1, max(ys))
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    clase_nombre = CLASSES[class_id]
    contador[clase_nombre] += 1
    filename = f"{clase_nombre}_{int(time.time()*1000)}"
    
    # Guardar archivos
    cv2.imwrite(str(img_path / f"{filename}.jpg"), frame)
    with open(lbl_path / f"{filename}.txt", "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

# -------------------------
# BUCLE DE CAPTURA
# -------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

        now = time.time()
        if now - last_capture > CAPTURE_DELAY and contador[CLASSES[clase_actual]] < MAX_IMAGES_PER_CLASS:
            save_yolo_data(frame, hand, clase_actual)
            last_capture = now

    # UI
    info = f"GRABANDO: {CLASSES[clase_actual].upper()} | " + " ".join([f"{c[0]}:{contador[c]}" for c in CLASSES])
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Captura YOLO Limpia", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    elif key in [ord("1"), ord("2"), ord("3")]: clase_actual = int(chr(key)) - 1

cap.release()
cv2.destroyAllWindows()
