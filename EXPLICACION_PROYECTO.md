# 📘 Manual de Explicación y Guía de Rutas del Proyecto

¡Hola Mateo! He preparado este documento explicativo detallado para que puedas consultar de manera rápida y sencilla **qué hace cada archivo**, **cómo funcionan las rutas de tu proyecto** y **a dónde debe dirigir cada una de ellas**. 

Puedes leer este archivo cuantas veces quieras directamente en tu editor de código.

---

## 📂 1. ¿Qué hacen los scripts principales? (Explicación Sencilla)

### 📸 `prueba_deteccion.py` (El Extractor de Imágenes)
* **¿Qué hace?**
  Abre uno por uno tus videos crudos de accidentes (`.mp4`) que están guardados en tu carpeta `archive/`. Va avanzando por el video y **guarda una foto fija (`.jpg`) cada 10 fotogramas (frames)**.
* **¿Para qué sirve?**
  Las redes neuronales como YOLO aprenden a través de imágenes estáticas (fotos), no de videos directamente. Este archivo automatiza el trabajo de "tomar capturas" para construir el banco de fotos que subirás a plataformas como *MakeSense.ai* para dibujar las cajas rojas del choque (anotación).

### 🚗 `prueba_final.py` (El Detector en Tiempo Real)
* **¿Qué hace?**
  Toma un video de prueba cualquiera, carga el "cerebro entrenado" de la IA (el archivo de pesos `best.pt` generado al entrenar) y **analiza el video frame por frame en tiempo real**.
* **¿Para qué sirve?**
  Es el resultado final del proyecto. Sirve para evaluar si la IA realmente aprendió a identificar choques. Si la IA detecta un accidente, dibuja de forma automática una **caja roja en la pantalla** alrededor de los autos colisionando y escribe la etiqueta `"accidente"`.

### 🔄 `reentrenar_modelo.py` (Re-entrenamiento Incremental)
* **¿Qué hace?**
  Toma los pesos de tu mejor entrenamiento previo (`runs/detect/mi_modelo_accidentes/weights/best.pt`) y continúa el proceso de aprendizaje incorporando las fotos y etiquetas nuevas.
* **¿Para qué sirve?**
  Permite expandir y mejorar tu modelo sin tener que empezar desde cero. La IA conserva todo lo aprendido antes y solo refina su conocimiento con las nuevas fotos.

---

## 📍 2. Guía Completa de Rutas en tu PC

En tu computadora local, la carpeta raíz de tu proyecto es:
`C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes`

A continuación, se detalla qué hace cada ruta configurada en el proyecto y a dónde dirige exactamente:

### A) En el archivo de configuración `dataset.yaml`
Este archivo es el mapa de carreteras que le indica a YOLO dónde buscar las fotos y etiquetas de los accidentes.

*   **`path`**: **Carpeta raíz del dataset.**
    *   *¿A dónde dirige?* A `C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/Data-Guardada`.
    *   *¿Por qué?* Tras limpiar las subcarpetas redundantes, ahora las categorías como `head_on_collision` están directamente bajo la carpeta raíz `Data-Guardada`.
*   **`train`**: **Imágenes para el entrenamiento.**
    *   *¿A dónde dirige?* A la subcarpeta `head_on_collision` (es una ruta relativa, por lo que YOLO la busca dentro del `path` definido arriba).
*   **`val`**: **Imágenes para validar el aprendizaje.**
    *   *¿A dónde dirige?* A `head_on_collision` (YOLO evalúa su precisión comparando con este set).

### B) En el script `entrenar_modelo.py`
Este archivo inicia el entrenamiento de la red neuronal desde cero.

*   **`data`**: **Ubicación del mapa de datos.**
    *   *¿A dónde dirige?* A `C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/dataset.yaml`.
    *   *¿Por qué?* Le dice al script de entrenamiento de Python en qué parte exacta de tu PC está el archivo YAML con las instrucciones del dataset.

### C) En el script `prueba_deteccion.py`
Este script prepara tu dataset extrayendo fotos desde los videos crudos.

*   **`carpeta_origen_videos`**: **Origen de videos crudos.**
    *   *¿A dónde dirige?* A `C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/archive/Video-Accident-Dataset/head_on_collision`.
    *   *¿Por qué?* Indica la carpeta donde descargaste y guardaste los videos originales de choques que quieres procesar.
*   **`carpeta_destino_principal`**: **Destino de imágenes fijas.**
    *   *¿A dónde dirige?* A `C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/Data-Guardada`.
    *   *¿Por qué?* Es la ubicación en tu disco local donde deseas que el script guarde las capturas `.jpg` organizadas por categorías.

### D) En el script `prueba_final.py`
Este archivo hace la demostración de la IA en tiempo real.

*   **`ruta_mi_modelo`**: **Los pesos de la red neuronal.**
    *   *¿A dónde dirige?* A `C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/runs/detect/mi_modelo_accidentes/weights/best.pt`.
    *   *¿Por qué?* Le indica a la librería Ultralytics (YOLO) dónde se guardó el "cerebro entrenado" localmente tras el proceso de aprendizaje.
*   **`ruta_video_prueba`**: **El video para evaluar.**
    *   *¿A dónde dirige?* A `C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/archive/Video-Accident-Dataset/head_on_collision/head_on_collision_25.mp4`.
    *   *¿Por qué?* Es el video real de prueba que cargará el script para mostrarte en vivo cómo trabaja tu IA.

---

## 🚀 3. ¿Cómo ejecutar el proyecto localmente?

Cuando estés listo para probar, abre una terminal en tu computadora dentro de la carpeta del proyecto y ejecuta estos comandos:

1.  **Para iniciar el entrenamiento inicial (desde cero):**
    ```bash
    py entrenar_modelo.py
    ```
    *Nota: Esto leerá el dataset y comenzará el aprendizaje desde la red neuronal base yolov8n.*

2.  **Para iniciar el re-entrenamiento con imágenes nuevas (Transfer Learning):**
    ```bash
    py reentrenar_modelo.py
    ```
    *Nota: Este script tomará tus pesos previos 'best.pt' y continuará el aprendizaje agregando tus nuevas anotaciones.*

3.  **Para probar los resultados de la IA en tiempo real (OpenCV):**
    ```bash
    py prueba_final.py
    ```
    *Nota: Se abrirá una ventana clásica de OpenCV mostrando las predicciones.*

4.  **Para lanzar tu Panel Interactivo Web (Streamlit Dashboard) Premium:**
    ```bash
    py -m streamlit run app_dashboard.py
    ```
    *Nota: Este comando inicia tu aplicación interactiva local y la abre en tu navegador de internet en tiempo real.*
