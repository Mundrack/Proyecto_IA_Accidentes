from ultralytics import YOLO

print("Iniciando el entrenamiento de la IA...")

# 1. Cargamos el modelo base (el cerebro vacío pero con conocimientos generales)
modelo = YOLO('yolov8n.pt')

# 2. Configuramos y arrancamos el entrenamiento
resultados = modelo.train(
    data='T:/WORK SPACE/UDLA/Proyecto_IA_Accidentes/dataset.yaml', # La ruta a tu mapa
    epochs=25,       # Cantidad de veces que la IA repasará todas tus fotos completas
    imgsz=640,       # Tamaño al que se redimensionarán las imágenes para aprender
    plots=True,      # ¡Clave! Le pedimos que genere gráficos y tablas de rendimiento automáticamente
    name='mi_modelo_accidentes' # Nombre de la carpeta donde guardará los resultados
)

print("¡Entrenamiento finalizado con éxito!")