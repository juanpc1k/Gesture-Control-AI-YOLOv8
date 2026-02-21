import cv2
from ultralytics import YOLO

# ==========================================
# 1. CARGA DEL MODELO (Prioridad a last.pt por si acaso)
# ==========================================
try:
    # Intentamos cargar last.pt primero ya que best.pt parece estar sesgado a la palma
    model = YOLO('runs/detect/modelo_manos_v1/weights/last.pt')
    print("✅ Cargado 'last.pt' (Estado más reciente del entrenamiento)")
except:
    try:
        model = YOLO('best.pt')
        print("✅ Cargado 'best.pt' desde carpeta raíz")
    except Exception as e:
        print(f"❌ ERROR: No se encuentran los pesos. Asegúrate de que el archivo .pt existe.\n{e}")
        exit()

# Verificación de clases
print(f"🧠 Clases detectadas por el modelo: {model.names}")

# ==========================================
# 2. MAPEO DE COMANDOS
# ==========================================
# Este diccionario traduce lo que ve la IA a lenguaje de drones
# Si tus clases tienen otros nombres, cámbialos aquí:
ACCIONES = {
    'fist': "🚀 [DESPEGUE]",
    'palm': "🛑 [ATERRIZAJE / EMERGENCIA]",
    'thumbs_up': "🔄 [PIRUETA / FLIP]"
}

# ==========================================
# 3. EJECUCIÓN EN TIEMPO REAL
# ==========================================
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 150) # Prueba a subir este valor (0-255)
ultimo_gesto = ""

print("\n--- INICIANDO CONTROL GESTUAL (Modo Sensible) ---")
print("Tips: Si no detecta, aléjate un poco de la cámara o mejora la luz.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Efecto espejo
    frame = cv2.flip(frame, 1)

    # INFERENCIA OPTIMIZADA
    # conf=0.25: Muy sensible para captar el puño y el pulgar
    # iou=0.45: Evita que se solapen cuadros si hay dudas
    results = model.predict(source=frame, conf=0.40, iou=0.45, show=False, verbose=False)

    # Dibujar resultados
    annotated_frame = results[0].plot()

    # PROCESAMIENTO DE LA DETECCIÓN
    if len(results[0].boxes) > 0:
        # Obtenemos la caja con mayor confianza
        mejor_box = results[0].boxes[0]
        cls_id = int(mejor_box.cls[0])
        conf_score = mejor_box.conf[0]
        nombre_gesto = model.names[cls_id]
        
        # Mostrar comando si cambia el gesto
        if nombre_gesto != ultimo_gesto:
            comando = ACCIONES.get(nombre_gesto, f"Gesto: {nombre_gesto}")
            print(f"{comando} (Confianza: {conf_score:.2f})")
            ultimo_gesto = nombre_gesto
    else:
        ultimo_gesto = ""

    # Interfaz de usuario
    cv2.putText(annotated_frame, f"Gesto actual: {ultimo_gesto}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('IADrones - Test de Control', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n--- TEST FINALIZADO ---")