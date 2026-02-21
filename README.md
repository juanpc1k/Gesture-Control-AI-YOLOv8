# 🤖 Gesture Control AI with YOLOv8 & MediaPipe  

![Python](https://img.shields.io/badge/Python-3.13-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-green)
![Torch](https://img.shields.io/badge/PyTorch-CPU-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

Sistema de reconocimiento de gestos manuales en tiempo real utilizando **YOLOv8** y **MediaPipe**, diseñado para el control gestual de drones, robótica e interfaces sin contacto.

---

## 🎯 Descripción General

Este proyecto implementa un pipeline completo de visión artificial:

1. 📷 Captura automática de datos  
2. 🏷️ Generación automática de etiquetas YOLO  
3. 🧠 Entrenamiento con YOLOv8 Nano  
4. ⚡ Inferencia en tiempo real  
5. 🎮 Traducción de gestos a comandos  

Optimizado para funcionar en **CPU**, sin necesidad de GPU.

---

## 🛠️ Características

### 🎯 Captura Inteligente
- Uso de MediaPipe Hand Landmarks  
- Generación automática de Bounding Boxes  
- Exportación directa en formato YOLO  

### ⚡ Modelo Optimizado
- YOLOv8 Nano  
- Alta tasa de FPS  
- Bajo consumo de recursos  
- Ideal para sistemas embebidos  

### 🧠 Sistema de Control

| Gesto | Acción |
|-------|--------|
| ✋ Palma abierta | Despegue |
| ✊ Puño cerrado | Aterrizaje |
| 👍 Pulgar arriba | Acción Especial |

---

## 📁 Estructura del Proyecto

```bash
gesture-control-ai/
│
├── dataset_hands/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   │
│   └── val/
│       ├── images/
│       └── labels/
│
├── capturar_dataset.py
├── entrenar_yolo.py
├── test_modelo.py
└── dataset.yaml
```

---

## 🧩 Retos Técnicos

### ❌ Problema

Python 3.13 no tenía versiones estables de PyTorch con soporte CUDA en el momento del desarrollo.

### ✅ Solución Implementada

- Migración estratégica a entorno CPU-only  
- Uso de distribuciones estables de Torch  
- Optimización mediante YOLOv8 Nano  
- Procesamiento multihilo en Intel 13ª Gen  

📌 Resultado: entrenamiento fluido e inferencia en tiempo real sin GPU.

---

## 🚀 Instalación

### 1️⃣ Clonar repositorio

```bash
git clone https://github.com/tuusuario/gesture-control-ai.git
cd gesture-control-ai
```

### 2️⃣ Instalar dependencias

```bash
pip install ultralytics mediapipe opencv-python torch torchvision
```

---

## 🧪 Flujo de Uso

### 🎥 1. Generar Dataset

```bash
python capturar_dataset.py
```

Detecta landmarks con MediaPipe y genera automáticamente las etiquetas YOLO.

---

### 🏋️ 2. Entrenar Modelo

```bash
python entrenar_yolo.py
```

Se generará:

- dataset.yaml  
- Carpeta runs/  
- Archivo best.pt  

---

### 🎮 3. Probar Inferencia

```bash
python test_modelo.py
```

Carga automática del modelo entrenado y detección en tiempo real.

---

## 📈 Resultados

### 🎯 Clases Entrenadas

| ID | Clase | Comando |
|----|-------|---------|
| 0 | Palma | Despegue |
| 1 | Puño | Aterrizaje |
| 2 | Pulgar Arriba | Acción Especial |

---

## 🧠 Arquitectura

- Modelo: YOLOv8 Nano  
- Tipo: CNN (Convolutional Neural Network)  
- Framework: Ultralytics  
- Hardware: CPU optimizado  

---

## 🎮 Aplicaciones Potenciales

- 🚁 Control gestual de drones (UAV)  
- 🤖 Robótica colaborativa  
- 🖥️ Interfaces sin contacto  
- 🕶️ Sistemas XR  
- 🏭 Automatización industrial  

---

## 📌 Futuras Mejoras

- 🔥 Versión con GPU (CUDA)  
- 📱 Implementación en Edge Devices  
- 🧠 Añadir más clases de gestos  
- 🎛️ Integración directa con control de drones reales  

---

## 👨‍💻 Autor

Proyecto desarrollado como sistema completo de visión artificial aplicada en tiempo real. JuanPC1k
