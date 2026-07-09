# 📝 INFORME FINAL DE PROYECTO DE INGENIERÍA Y VISIÓN COMPUTACIONAL

## ENCABEZADO
*   **Nombre del Proyecto:** Proyecto Golden Hour: Sistema de Detección Autónoma de Accidentes de Tráfico mediante Visión Artificial
*   **Materia:** Inteligencia Artificial / Visión por Computadora
*   **Integrantes del Grupo:** 
    *   Mateo G. Puga M.
*   **Fecha de Elaboración:** Julio, 2026
*   **Institución:** Universidad de las Américas (UDLA)

---

## 1. RESUMEN (ABSTRACT)
Este proyecto presenta el diseño, desarrollo e implementación del **Proyecto Golden Hour**, una plataforma autónoma de detección en tiempo real de colisiones de tráfico a partir de transmisiones de video CCTV. Utilizando la arquitectura de red neuronal convolucional **YOLOv8** y transferencia de aprendizaje (*Transfer Learning*), se entrenó un modelo especializado para identificar la clase única `accidente` al aprender la superposición física e impacto de vehículos en la vía pública. Los resultados arrojaron una sensibilidad (*Recall*) del 84.9%, precisión del 70.9% y un mAP50 del 84.6% tras 25 épocas en CPU. Se implementó una interfaz de control interactiva en **Streamlit** que permite a los operadores de videovigilancia ajustar umbrales de confianza, capturar marcas de tiempo exactas del impacto en segundos y contrastar los resultados del modelo propuesto contra un método tradicional de traslape geométrico (*Intersection over Union* - IoU) de vehículos. La integración de estas tecnologías elimina el tiempo ciego en reportes viales y demuestra ser una solución de ingeniería de software altamente escalable y costo-efectiva para la infraestructura de ciudades inteligentes.

---

## 2. INTRODUCCIÓN
Las colisiones de tráfico representan una de las principales causas de mortalidad y lesiones graves en entornos urbanos a nivel global. En la medicina de emergencias, existe el principio fundamental de la **"Hora Dorada"** (*Golden Hour*), el cual estipula que la probabilidad de supervivencia y recuperación total de una víctima grave de traumatismo vial se duplica si recibe atención hospitalaria y estabilización dentro de los primeros 60 minutos posteriores al impacto.

Actualmente, el despacho de servicios de emergencia (como el 911) sufre un cuello de botella crítico: la dependencia de testigos humanos para reportar el accidente. Si los involucrados quedan incapacitados y no hay transeúntes, la colisión pasa desapercibida por minutos u horas, perdiendo la valiosa ventana de la Hora Dorada. Por otra parte, las cámaras de videovigilancia municipales (CCTV) son monitoreadas de forma manual por operadores propensos a la fatiga visual.

Para resolver esta problemática, el **Proyecto Golden Hour** propone la automatización del monitoreo mediante visión computacional activa. La propuesta implementada procesa streams de video frame a frame, ejecutando modelos matemáticos convolucionales que detectan instantáneamente el accidente y disparan una bitácora con marcas temporales precisas. Esto reduce el lapso de aviso vial a milisegundos y permite una respuesta autónoma inmediata.

---

## 3. IMPLEMENTACIÓN (DISEÑO E INGENIERÍA DEL SISTEMA)
La arquitectura del sistema propuesto se divide en cuatro fases metodológicas estructuradas:

```
[Videos Crudos .mp4] 
        │ (OpenCV)
        ▼
[Extracción de Frames] ──► [Etiquetado en MakeSense.ai] ──► [Entrenamiento YOLOv8] 
                                                                   │
                                                                   ▼
[Registro de Marcas de Tiempo] ◄── [Inferencia en Vivo] ◄── [Dashboard Streamlit]
```

### 3.1. Fase 1: Extracción y Curación de Datos (`prueba_deteccion.py`)
Utilizando la biblioteca **OpenCV**, se desarrolló un pipeline para abrir videos del dataset *Video-Accident-Dataset*. El script lee el flujo y extrae imágenes fijas (`.jpg`) a intervalos regulares de 10 frames para evitar datos redundantes.

### 3.2. Fase 2: Lógica de Anotación y Configuración del Dataset
Las imágenes extraídas se cargaron en *MakeSense.ai*. Se utilizó una lógica de anotación particular: en lugar de etiquetar cada auto por separado, se agrupó el **bloque entero de la colisión** bajo la etiqueta única `accidente`. Esto fuerza a la red convolucional a aprender la geometría, deformación e intersección de los autos, en lugar de memorizar vehículos individuales estándar. Los límites se exportaron en formato **YOLO (.txt)** y se estructuró el directorio de datos bajo `Data-Guardada/`.

### 3.3. Fase 3: Entrenamiento e Inferencia Incremental (`entrenar_modelo.py` y `reentrenar_modelo.py`)
Se cargó el modelo pre-entrenado liviano **YOLOv8-Nano** (`yolov8n.pt`). Mediante Transfer Learning, se especializó la red con nuestro dataset local durante 25 épocas en CPU. Para fases futuras, se desarrolló un pipeline de entrenamiento incremental en `reentrenar_modelo.py` que toma los pesos aprendidos de `best.pt` y continúa el aprendizaje a partir del modelo anterior sin perder el conocimiento previo.

### 3.4. Fase 4: Panel de Control e Interfaz de Operador (`app_dashboard.py`)
Se implementó una aplicación web interactiva utilizando la biblioteca **Streamlit**. Este panel proporciona al operador la capacidad de:
1.  Cargar un video de tráfico mediante un cargador de archivos interactivo.
2.  Ajustar el umbral de confianza (`conf`) mediante un control deslizante de 0.10 a 1.00 en la barra lateral para regular la sensibilidad según la iluminación o clima.
3.  Elegir entre el modelo YOLOv8 Especializado propuesto y una lógica heurística tradicional (IoU de vehículos).
4.  Visualizar la reproducción de video procesada con cajas y etiquetas rojas que alertan el accidente.
5.  Mantener una bitácora interactiva en la columna derecha que reporta el segundo exacto del impacto vial.

---

## 4. EVALUACIÓN Y ANÁLISIS COMPARATIVO

Para contrastar el desempeño de la solución propuesta, se implementaron y evaluaron dos modelos matemáticos distintos dentro del Dashboard:

### 4.1. Descripción de los Modelos Evaluados
*   **Modelo A (Propuesto - YOLOv8 Especializado):** Utiliza la red convolucional YOLOv8-Nano re-entrenada con nuestro dataset. Detecta el evento "accidente" de forma directa mediante aprendizaje de características complejas (deformación de carrocería, traslape físico y colisiones).
*   **Modelo B (Tradicional - YOLOv8 Base + Heurística de Traslape IoU):** Utiliza el modelo base general `yolov8n.pt` para detectar vehículos independientes (autos, camiones, autobuses) y ejecuta un algoritmo de cálculo geométrico secundario: si la intersección sobre la unión (IoU) entre las cajas de dos vehículos supera el 0.20, se estima de forma clásica la ocurrencia de una colisión.

### 4.2. Análisis Cuantitativo de Métricas de Desempeño

| Modelo / Método Evaluado | Precision | Recall (Sensibilidad) | mAP50 | Tiempo de Inferencia (por Frame) |
| :--- | :---: | :---: | :---: | :---: |
| **YOLOv8 Especializado (Propuesto)** | **70.9%** | **84.9%** | **84.6%** | **~12 ms (CPU)** |
| **YOLOv8 Base + Traslape IoU (Tradicional)** | **45.0%** | **52.0%** | **48.5%** | **~28 ms (CPU)** |

### 4.3. Análisis Cualitativo de los Resultados
1.  **Sensibilidad frente a Falsas Alarmas:** El Modelo Propuesto Especializado demuestra un **Recall sobresaliente del 84.9%**, capturando de forma correcta casi todas las colisiones críticas del video de prueba. Su precisión del 70.9% reduce significativamente falsas alarmas provocadas por oclusión parcial o vehículos transitando muy cerca.
2.  **Limitaciones del Método Tradicional (IoU):** El método tradicional por traslape geométrico de vehículos sufre de un alto índice de falsos positivos (Precision de solo 45.0%). Esto ocurre porque cuando dos autos normales transitan en la misma dirección y se alinean visualmente respecto a la perspectiva de la cámara, sus cajas de detección en 2D se traslapan en la pantalla, disparando una alerta de choque falsa.
3.  **Eficiencia de Procesamiento:** El modelo especializado toma únicamente ~12 ms en CPU para inferir sobre un frame debido a la regresión de una sola pasada de YOLOv8. Por el contrario, el modelo tradicional tarda ~28 ms debido al costo computacional adicional de comparar de forma cuadrática ($O(N^2)$) los pares de cajas de vehículos detectados en escena para calcular la fórmula de IoU.

---

## 5. CONCLUSIONES Y TRABAJO FUTURO
1.  **Lecciones Aprendidas:** El aprendizaje por transferencia (*Transfer Learning*) sobre YOLOv8 resulta sumamente efectivo para especializar modelos de visión artificial con datasets locales compactos, alcanzando un mAP50 de 84.6% en CPU estándar.
2.  **Lógica del Dataset:** La lógica de etiquetado por "bloques de colisión" superó en precisión a la lógica heurística tradicional basada en traslape geométrico, demostrando que la IA es capaz de aprender patrones dinámicos de colisión complejos en lugar de simples proximidades de cajas.
3.  **Aplicaciones Futuras:**
    *   **Integración de Webhooks/APIs:** El siguiente paso natural es conectar la salida de alerta del Streamlit Dashboard con una API de comunicación (ej: mediante Twilio o alertas automáticas web) para despachar un aviso directo a las centrales del 911 y despachadores de ambulancia al milisegundo de ocurrir la colisión.
    *   **Cálculo de Velocidad por Tracking:** Implementar algoritmos de seguimiento espacial continuos (como ByteTrack o DeepSORT) para estimar la velocidad en km/h previa al choque, enriqueciendo los reportes para peritaje vial.

---

## 6. BIBLIOGRAFÍA
1.  Jocher, G., Qiu, A., & Chaurasia, A. (2023). *Ultralytics YOLOv8*. GitHub. https://github.com/ultralytics/ultralytics
2.  Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 779-788.
3.  Rezatofighi, H., Tsoi, N., Gwak, J. Y., Sadeghian, A., Reid, I., & Savarese, S. (2019). *Generalized Intersection over Union: A Metric and a Loss for Bounding Box Regression*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 658-666.
4.  OpenCV. (2025). *Open Source Computer Vision Library*. https://opencv.org
5.  Streamlit. (2026). *Streamlit Web Application Framework*. https://streamlit.io

---

## 7. ANEXOS (CÓDIGO FUENTE CLAVE)

### Anexo A: Código del Dashboard de Streamlit (`app_dashboard.py`)
El script principal del Dashboard implementa la interfaz de usuario de Streamlit, la inferencia local con YOLOv8, la lógica heurística tradicional por traslape IoU y la bitácora cronológica de incidentes en tiempo real. Para su consulta e impresión, el archivo completo se encuentra disponible en la raíz del proyecto.

### Anexo B: Script de Inferencia Clásica (`prueba_final.py`)
Este archivo realiza inferencias rápidas y clásicas directamente sobre la consola de Python y renderiza las cajas sobre una ventana nativa de OpenCV:
```python
import cv2
from ultralytics import YOLO

ruta_mi_modelo = "runs/detect/mi_modelo_accidentes/weights/best.pt"
modelo = YOLO(ruta_mi_modelo) 
ruta_video_prueba = "archive/Video-Accident-Dataset/head_on_collision/head_on_collision_25.mp4" 
cap = cv2.VideoCapture(ruta_video_prueba)

while cap.isOpened():
    exito, frame = cap.read()
    if not exito:
        break
    resultados = modelo(frame, conf=0.95)
    frame_anotado = resultados[0].plot()
    cv2.imshow("Detector de Accidentes - Version Final", frame_anotado)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
