import cv2
from ultralytics import YOLO
import winsound  # Librería estándar de Windows para sonidos

# ==========================================
# 1. CARGA DEL MODELO
# ==========================================
model = YOLO('best.pt')

# ==========================================
# 2. CONFIGURACIÓN Y COMANDOS
# ==========================================
ACCIONES = {
    'palm': {"msg": "🚀 [DESPEGUE]", "frec": 1000, "dur": 200},
    'fist': {"msg": "🛑 [ATERRIZAJE / EMERGENCIA]", "frec": 400, "dur": 600},
    'thumbs_up': {"msg": "🔄 [PIRUETA / FLIP]", "frec": 1500, "dur": 150}
}

contador_gestos = {nombre: 0 for nombre in model.names.values()}
UMBRAL_ESTABILIDAD = 3 
ultimo_comando_confirmado = ""

# ==========================================
# 3. BUCLE PRINCIPAL
# ==========================================
cap = cv2.VideoCapture(0)
print("\n--- SISTEMA GESTUAL CON FEEDBACK SONORO ACTIVO ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    # Inferencia con el "Modo Rescate" (conf=0.15) que nos funcionó
    results = model.predict(source=frame, conf=0.15, iou=0.45, verbose=False)
    annotated_frame = results[0].plot()
    
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        nombre_gesto = model.names[int(box.cls[0])]
        conf_score = float(box.conf[0])

        # Lógica de Estabilidad
        for gesto in contador_gestos:
            if gesto == nombre_gesto:
                contador_gestos[gesto] += 1
            else:
                contador_gestos[gesto] = 0

        # Si el gesto es estable, lanzamos acción y sonido
        if contador_gestos[nombre_gesto] >= UMBRAL_ESTABILIDAD:
            if nombre_gesto != ultimo_comando_confirmado:
                info = ACCIONES.get(nombre_gesto)
                print(f"{info['msg']} (Conf: {conf_score:.2f}) ✅")
                
                # Feedback sonoro (Frecuencia, Duración)
                winsound.Beep(info['frec'], info['dur'])
                
                ultimo_comando_confirmado = nombre_gesto
    else:
        for g in contador_gestos: contador_gestos[g] = 0

    # Interfaz en pantalla
    cv2.putText(annotated_frame, f"COMANDO: {ultimo_comando_confirmado}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    cv2.imshow('DRONE CONTROL PRO', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()