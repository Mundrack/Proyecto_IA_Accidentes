import cv2
from ultralytics import YOLO

# 1. Cargamos TU modelo recién entrenado (Asegúrate de que la ruta sea correcta)
ruta_mi_modelo = "T:/WORK SPACE/UDLA/Proyecto_IA_Accidentes/runs/detect/mi_modelo_accidentes/weights/best.pt"
modelo = YOLO(ruta_mi_modelo) 

# 2. Ruta a un video de prueba (Cámbiala por un video que la IA no haya visto)
ruta_video_prueba = "T:/WORK SPACE/UDLA/Proyecto_IA_Accidentes/Video-Accident-Dataset/head_on_collision/head_on_collision_5.mp4" 
cap = cv2.VideoCapture(ruta_video_prueba)

print("Iniciando prueba de detección...")

while cap.isOpened():
    exito, frame = cap.read()
    if not exito:
        print("Fin del video.")
        break

    # 3. Tu IA analiza el frame buscando la clase "accidente"
    resultados = modelo(frame)

    # 4. Dibujamos la caja roja si detecta el choque
    frame_anotado = resultados[0].plot()

    # 5. Mostramos el video
    cv2.imshow("Detector de Accidentes - Version Final", frame_anotado)

    # Tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()