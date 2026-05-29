from ultralytics import YOLO

print("Iniciando el entrenamiento de la IA...")

# 1. Cargamos el modelo base (el cerebro vacío pero con conocimientos generales)
modelo = YOLO('yolov8n.pt')

# 2. Configuramos y arrancamos el entrenamiento
resultados = modelo.train(
    # data: La ruta absoluta local al archivo 'dataset.yaml' en tu computadora.
    # Este archivo de configuración le enseña a YOLO dónde buscar las imágenes y las etiquetas reales.
    data='C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/dataset.yaml', # Mapa local de datos
    epochs=25,       # Cantidad de veces que la IA repasará todas tus fotos completas
    imgsz=640,       # Tamaño al que se redimensionarán las imágenes para aprender
    plots=True,      # ¡Clave! Le pedimos que genere gráficos y tablas de rendimiento automáticamente
    name='mi_modelo_accidentes' # Nombre de la carpeta donde guardará los resultados
)

print("¡Entrenamiento finalizado con éxito!")