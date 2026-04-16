import cv2
import os
import glob

# --- CONFIGURACIÓN ---
# La carpeta de origen que tiene los videos (ej. head_on_collision)
carpeta_origen_videos = "T:/WORK SPACE/UDLA/Proyecto_IA_Accidentes/Video-Accident-Dataset/side_collision"

# La carpeta principal donde se guardará todo
carpeta_destino_principal = "T:/WORK SPACE/UDLA/Proyecto_IA_Accidentes/Data-Guardada"

# ¿Cuántos videos máximo quieres procesar de esta carpeta?
limite_videos = 30

# Guardar un frame cada X frames
intervalo_frames = 10 
# ---------------------

# 1. Obtener el nombre de la categoría (ej. "head_on_collision") a partir de la ruta
nombre_categoria = os.path.basename(os.path.normpath(carpeta_origen_videos))

# 2. Crear la subcarpeta de destino específica para esta categoría
ruta_destino_categoria = os.path.join(carpeta_destino_principal, nombre_categoria)
if not os.path.exists(ruta_destino_categoria):
    os.makedirs(ruta_destino_categoria)

# 3. Buscar todos los archivos .mp4 en la carpeta de origen
# (Usamos glob para encontrar los archivos fácilmente)
patron_busqueda = os.path.join(carpeta_origen_videos, "*.mp4")
lista_videos = glob.glob(patron_busqueda)

print(f"Se encontraron {len(lista_videos)} videos en '{nombre_categoria}'. Procesando un máximo de {limite_videos}...")

# 4. Procesar los videos hasta llegar al límite
videos_procesados = 0

for ruta_video in lista_videos:
    if videos_procesados >= limite_videos:
        break
    
    # Obtener el nombre del video (ej. "head_on_collision_123.mp4")
    nombre_video = os.path.basename(ruta_video)
    # Quitarle la extensión para usarlo en el nombre de la imagen
    nombre_base = os.path.splitext(nombre_video)[0] 

    cap = cv2.VideoCapture(ruta_video)
    contador_frames = 0
    imagenes_guardadas_este_video = 0

    print(f"  -> Extrayendo de: {nombre_video}")

    while cap.isOpened():
        exito, frame = cap.read()
        if not exito:
            break
        
        if contador_frames % intervalo_frames == 0:
            # Crear un nombre único: nombreVideo_frameX.jpg
            nombre_archivo = f"{nombre_base}_frame_{contador_frames}.jpg"
            ruta_guardado = os.path.join(ruta_destino_categoria, nombre_archivo)
            
            cv2.imwrite(ruta_guardado, frame)
            imagenes_guardadas_este_video += 1
            
        contador_frames += 1

    cap.release()
    videos_procesados += 1

print(f"\n¡Proceso finalizado para la categoría '{nombre_categoria}'!")
print(f"Revisa la carpeta: {ruta_destino_categoria}")