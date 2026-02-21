# 📖 Guía de Uso y Especificaciones Técnicas

Este proyecto documenta el desarrollo de un sistema de reconocimiento de gestos mediante **YOLOv8**, optimizado para ejecutarse en entornos modernos y controlado mediante hardware estándar.

---

## 💻 Entorno de Ejecución y Hardware

El proyecto ha sido desarrollado y testeado íntegramente en **Windows**, lo que ofrece ventajas específicas para este pipeline:
* **Acceso Directo a Periféricos:** Uso de la API nativa de Windows para una integración inmediata con la **Webcam** mediante OpenCV, permitiendo una latencia mínima en la captura y previsualización.
* **Optimización de CPU:** Entrenamiento ejecutado sobre un procesador **Intel Core i5 de 13ª Gen**, aprovechando su arquitectura multihilo para suplir la falta de compatibilidad temporal de drivers GPU en versiones recientes de Python.
* **Aislamiento:** Uso de entornos virtuales (`venv`) para garantizar la estabilidad de las librerías.

---

## 🛠️ Stack Tecnológico (Versiones Utilizadas)

Para asegurar la replicabilidad del proyecto en **Python 3.13**, se deben utilizar las siguientes versiones:

| Biblioteca | Versión | Función |
| :--- | :--- | :--- |
| **Python** | 3.13.1 | Lenguaje principal |
| **Ultralytics** | 8.3.x | Engine de YOLOv8 (Entrenamiento e Inferencia) |
| **Torch** | 2.6.0+cpu | Motor de tensores optimizado para CPU |
| **Torchvision** | 0.21.0+cpu | Utilidades para visión artificial |
| **Mediapipe** | 0.10.20 | Extracción de landmarks para auto-etiquetado |
| **OpenCV-Python** | 4.11.0.86 | Gestión de video y UI en tiempo real |

---

## 🏗️ Descripción de los Scripts

### 1. `grabar_hands_todo_en_uno.py`
Es el script de **Adquisición de Datos y Auto-etiquetado**:
* Utiliza **MediaPipe** para localizar la mano y sus 21 puntos clave.
* Genera automáticamente las cajas delimitadoras (*Bounding Boxes*) y exporta los archivos `.txt` en formato YOLO.
* Organiza los datos automáticamente en carpetas de `train` y `val` (ratio 80/20).
* **Requisito:** Requiere el archivo `hand_landmarker.task` en el directorio raíz.

### 2. `entrenar_yolo.py`
Script de **Entrenamiento del Modelo**:
* Genera de forma automatizada el archivo `dataset.yaml` con las rutas del sistema.
* Realiza *Transfer Learning* sobre el modelo base `yolov8n.pt`.
* Configurado para ejecutar el entrenamiento en CPU con monitorización de métricas de precisión (mAP50).

### 3. `test_modelo.py`
Script de **Inferencia y Control**:
* Carga los pesos entrenados (`best.pt`).
* Traduce las detecciones visuales en comandos lógicos impresos en consola (ej: "Despegue", "Pirueta").
* Implementa un filtro de confianza para evitar falsos positivos en el control.

---

## 📋 Requisitos de Archivos
Asegúrate de que tu estructura de carpetas contenga:
* `hand_landmarker.task` (Modelo base de MediaPipe).
* `dataset_hands/` (Generado automáticamente tras la captura).
* `dataset.yaml` (Generado automáticamente al entrenar).

---
Desarrollado como parte del proyecto de formación en **IA y Drones**.
