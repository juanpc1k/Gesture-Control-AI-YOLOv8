from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. DEFINIR RUTAS
    # Usamos barras normales / para evitar problemas de escape en Windows
    base_path = "C:/iadrones/semana5"
    dataset_yaml = f"{base_path}/dataset.yaml"
    
    # 2. CREAR EL ARCHIVO YAML DESDE AQUÍ MISMO
    # Así nos aseguramos de que existe y no tiene extensiones ocultas
    yaml_content = f"""
path: {base_path}/dataset_hands
train: train/images
val: val/images

names:
  0: palm
  1: fist
  2: thumbs_up
"""
    
    with open(dataset_yaml, "w") as f:
        f.write(yaml_content.strip())
    
    print(f"✅ Archivo YAML creado o actualizado en: {dataset_yaml}")

    # 3. VERIFICACIÓN MANUAL ANTES DE SEGUIR
    if os.path.exists(dataset_yaml):
        print("🚀 El archivo existe. Iniciando entrenamiento...")
        
        # 4. CARGAR MODELO Y ENTRENAR
        model = YOLO('yolov8n.pt')
        model.train(
            data=dataset_yaml,
            epochs=50,
            imgsz=640,
            batch=8,       # Bajamos un poco el batch para la CPU
            device='cpu',  # <--- CAMBIA ESTO A 'cpu'
            name='modelo_manos_v1'
        )
    else:
        print("❌ Error crítico: Python no pudo crear el archivo. Revisa permisos de carpeta.")
