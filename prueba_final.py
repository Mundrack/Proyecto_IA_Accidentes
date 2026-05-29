import cv2
from ultralytics import YOLO

# 1. Cargamos TU modelo recién entrenado
# ruta_mi_modelo: Ruta absoluta local al archivo de pesos 'best.pt' generados tras entrenar el modelo.
# Dirige a la carpeta 'runs/detect/mi_modelo_accidentes/weights/best.pt' en tu PC, que es donde se guarda el cerebro optimizado.
ruta_mi_modelo = "C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/runs/detect/mi_modelo_accidentes/weights/best.pt"
modelo = YOLO(ruta_mi_modelo) 

# 2. Ruta a un video de prueba
# ruta_video_prueba: Ruta absoluta en tu computadora al video (.mp4) que usarás para evaluar el modelo.
# Debe dirigir a un video real (por ejemplo, de la carpeta archive/) para que la IA realice la detección visual.
ruta_video_prueba = "C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/archive/Video-Accident-Dataset/head_on_collision/head_on_collision_25.mp4" 
cap = cv2.VideoCapture(ruta_video_prueba)

print("Iniciando prueba de detección...")

while cap.isOpened():
    exito, frame = cap.read()
    if not exito:
        print("Fin del video.")
        break

    # 3. Tu IA analiza el frame buscando la clase "accidente"
    # Hemos configurado 'conf=0.95' para que solo dibuje la caja roja si está al menos 95% segura del choque.
    resultados = modelo(frame, conf=0.95)

    # 4. Dibujamos la caja roja si detecta el choque
    frame_anotado = resultados[0].plot()

    # 5. Mostramos el video
    cv2.imshow("Detector de Accidentes - Version Final", frame_anotado)

    # Tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()