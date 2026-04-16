# Proyecto de Detección de Accidentes con IA

Este proyecto implementa un sistema para la detección automática de accidentes de tráfico utilizando **YOLOv8** (You Only Look Once), aprovechando redes neuronales para analizar video en tiempo real frame por frame.

## Arquitectura del Proyecto

El flujo de trabajo se divide en 3 etapas, cada una con su propio script de Python:

### 1. Extracción y Preparación de Datos (`prueba_deteccion.py`)
Toma videos de una carpeta origen (`Video-Accident-Dataset`), extrae imágenes fijas (frames) usando un intervalo definido y las clasifica. Este paso es esencial para crear nuestro banco de imágenes que usaremos para enseñar al modelo.
- Utiliza **OpenCV** para el manejo de video.

### 2. Entrenamiento del Modelo (`entrenar_modelo.py`)
Toma el dataset preparado (basado en la configuración de `dataset.yaml`) y comienza a entrenar partiendo del modelo base ligero de YOLO (`yolov8n.pt`).
- El proceso ocurre por 25 épocas usando imágenes de tamaño 640.
- Los resultados de gráficos y matrices de confusión se guardan en el directorio `runs/`.

### 3. Prueba e Inferencia (`prueba_final.py`)
Utiliza los pesos entrenados del modelo (los cuales se encuentran bajo `runs/detect/mi_modelo_accidentes/weights/best.pt`) para hacer predicciones en videos nunca antes vistos por la red neuronal. 
- Muestra el video procesado en pantalla con cajas de detección (bounding boxes).

## Requisitos de Instalación
Es necesario tener instalado Python junto a las siguientes bibliotecas:
```bash
pip install ultralytics opencv-python
```

## Cómo Utilizar

1. **Preparar Datos:** Asegúrate de colocar tus videos y ejecutar el script para la extracción de frames:
   ```bash
   python prueba_deteccion.py
   ```
2. **Entrenar:** Inicia el entrenamiento de la red neuronal:
   ```bash
   python entrenar_modelo.py
   ```
3. **Detección e Inferencia:** Comprueba los resultados del modelo entrenado sobre nuevos videos:
   ```bash
   python prueba_final.py
   ```

## Notas adicionales
- Los datasets originales, los videos pesados y los pesos del modelo han sido configurados en el `.gitignore` para no sobrecargar el repositorio de Git. 
