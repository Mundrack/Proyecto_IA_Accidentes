from ultralytics import YOLO

print("Iniciando el re-entrenamiento de la IA (Entrenamiento Incremental)...")

# 1. Cargamos TU modelo ya entrenado (Transfer Learning)
# Usamos la ruta absoluta al archivo 'best.pt' para evitar cualquier error de ejecución en Windows.
ruta_base_modelo = 'C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/runs/detect/mi_modelo_accidentes/weights/best.pt'
modelo = YOLO(ruta_base_modelo)

# 2. Configuramos y arrancamos el entrenamiento incremental
resultados = modelo.train(
    # data: Ruta absoluta local al archivo de configuración 'dataset.yaml'
    data='C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/dataset.yaml', 
    epochs=25,                       # Vueltas de aprendizaje (puedes aumentarlo si tienes más imágenes)
    imgsz=640,                       # Tamaño al que se redimensionarán las imágenes para aprender
    plots=True,                      # Generar tablas y gráficas de rendimiento automáticamente
    name='mi_modelo_accidentes_v2'   # Crea una nueva carpeta en 'runs/detect/' para no sobreescribir la v1
)

print("¡Re-entrenamiento finalizado con éxito! Los nuevos pesos están guardados en runs/detect/mi_modelo_accidentes_v2/")
