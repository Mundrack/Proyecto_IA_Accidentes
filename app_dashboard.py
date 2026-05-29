import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO

# =====================================================================
# CONFIGURACIÓN DE LA PÁGINA (ESTILO PREMIUM)
# =====================================================================
st.set_page_config(
    page_title="Sistema Automatizado de Detección de Accidentes",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de estilos CSS personalizados para lograr una estética premium (Neon Emerald/Slate Dark)
st.markdown("""
    <style>
        /* Tipografías y diseño general */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 42px;
            font-weight: 700;
            color: #10b981;
            text-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
            margin-bottom: 5px;
        }
        
        .subtitle {
            font-size: 18px;
            color: #94a3b8;
            margin-bottom: 25px;
        }
        
        /* Contenedores personalizados de alertas */
        .alert-card {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 5px solid #ef4444;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .alert-title {
            color: #ef4444;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 4px;
        }
        
        .alert-time {
            color: #f8fafc;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# ÁREA DE ENCABEZADO (MAIN HEADER)
# =====================================================================
st.markdown('<div class="main-title">🚨 Sistema Automatizado de Detección de Accidentes</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Visión computacional avanzada y analítica de tráfico en tiempo real para optimizar los tiempos de respuesta ante emergencias viales.</div>', unsafe_allow_html=True)

st.divider()

# =====================================================================
# BARRA LATERAL DE CONFIGURACIÓN (SIDEBAR)
# =====================================================================
st.sidebar.image("https://img.icons8.com/nolan/96/automatic-car-wash.png", width=70)
st.sidebar.markdown("### 🛠️ Configuración del Sistema")
st.sidebar.write("Ajusta los parámetros de inferencia de la Inteligencia Artificial:")

# Entrada de la ruta del modelo con valor por defecto
ruta_modelo_defecto = "C:/Users/MateoGPugaM/OneDrive - Grupo Radical/Escritorio/Proyecto_IA_Accidentes/runs/detect/mi_modelo_accidentes/weights/best.pt"
ruta_modelo = st.sidebar.text_input(
    "Ruta local del Modelo YOLO (.pt)",
    value=ruta_modelo_defecto,
    help="Especifica la ubicación absoluta de tu archivo de pesos entrenado best.pt"
)

# Slider para ajustar el umbral de confianza (sensibilidad de la IA)
sensibilidad = st.sidebar.slider(
    "Sensibilidad de la IA (Confianza)",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05,
    help="Umbral mínimo de seguridad que requiere el modelo para registrar y marcar un choque vial."
)

st.sidebar.divider()
st.sidebar.markdown("""
    **💡 Consejo de Operación:**
    * Un umbral **bajo (0.15 - 0.30)** es muy sensible y detecta choques lejanos o pequeños, pero es propenso a falsas alarmas.
    * Un umbral **alto (0.75 - 0.95)** es extremadamente seguro antes de alertar, pero podría ignorar impactos menores.
""")

# =====================================================================
# CARGADOR DE ARCHIVOS Y ESTRUCTURA DE LA PÁGINA principal
# =====================================================================
uploaded_file = st.file_uploader(
    "📂 Carga un archivo de video para iniciar el análisis", 
    type=["mp4", "avi", "mov"],
    help="Formatos de video soportados: .mp4, .avi, .mov"
)

# Si hay un video cargado, mostramos el botón de acción principal
if uploaded_file is not None:
    st.success("¡Video cargado con éxito en memoria! Listo para el análisis.")
    btn_iniciar = st.button("🚀 Iniciar Análisis de Inteligencia Artificial", use_container_width=True)
    
    if btn_iniciar:
        # Creamos las dos columnas para el procesamiento en tiempo real
        col1, col2 = st.columns([2, 1])
        
        # Columna 1 (Panel de Inferencia en Vivo)
        with col1:
            st.markdown("### 📺 Monitoreo de Video en Tiempo Real")
            frame_placeholder = st.empty()
            progreso_bar = st.progress(0)
            
        # Columna 2 (Reporte e Historial de Incidentes)
        with col2:
            st.markdown("### 📋 Historial de Incidentes Detectados")
            alertas_container = st.empty()
            
            # Estado inicial de las alertas
            alertas_container.info("Monitoreando... No se registran colisiones hasta el momento. ✅")
        
        # Guardar el video cargado en un archivo temporal de forma segura
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name
            
        # Cargar el modelo YOLO
        try:
            modelo = YOLO(ruta_modelo)
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo YOLO en la ruta provista: {e}")
            st.stop()
            
        # Lectura de video mediante OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if fps == 0 or total_frames == 0:
            fps = 30.0  # Valor por defecto seguro en caso de fallo al leer metadatos
            total_frames = 300
            
        # Variables de control de alertas
        eventos_accidentes = []
        last_alert_time = None  # Guardará el segundo del último accidente para agruparlo y evitar duplicados
        frame_counter = 0
        
        # Bucle de inferencia frame a frame
        while cap.isOpened():
            exito, frame = cap.read()
            if not exito:
                break
                
            frame_counter += 1
            segundo_actual = frame_counter / fps
            
            # Ejecutar la detección del modelo YOLO con la confianza seleccionada
            resultados = modelo(frame, conf=sensibilidad, verbose=False)
            
            # Dibujar las cajas rojas e información del modelo en el frame
            frame_anotado = resultados[0].plot()
            
            # Convertir formato BGR (OpenCV) a RGB (Streamlit)
            frame_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
            
            # Renderizar el frame procesado en el contenedor de la Columna 1
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Actualizar barra de progreso del video
            progreso = int((frame_counter / total_frames) * 100)
            progreso_bar.progress(min(progreso, 100))
            
            # Lógica de detección de colisiones
            if len(resultados[0].boxes) > 0:
                # Evitamos llenar la interfaz agrupando alertas consecutivas en una ventana de 3 segundos
                if last_alert_time is None or (segundo_actual - last_alert_time) > 3.0:
                    eventos_accidentes.append(segundo_actual)
                    last_alert_time = segundo_actual
                    
                    # Emitir aviso sonoro o alerta crítica en pantalla en tiempo real (Columna 2)
                    with alertas_container.container():
                        for ev in eventos_accidentes:
                            st.markdown(f"""
                                <div class="alert-card">
                                    <div class="alert-title">🚨 COLISIÓN DETECTADA</div>
                                    <div class="alert-time">Impacto registrado en el <b>segundo {ev:.2f}</b> del video.</div>
                                </div>
                            """, unsafe_allow_html=True)
            
            # Pequeño retardo para simular la velocidad real de reproducción de video
            time.sleep(0.01)
            
        # Liberar recursos de OpenCV y eliminar el archivo temporal
        cap.release()
        try:
            os.remove(temp_video_path)
        except Exception:
            pass
            
        # =====================================================================
        # PANEL FINAL DE RESULTADOS
        # =====================================================================
        st.divider()
        st.markdown("### 📊 Resumen Estadístico Final del Análisis")
        
        if not eventos_accidentes:
            st.balloons()
            st.success("💚 **Análisis Concluido sin Novedad:** El tráfico fluyó con total normalidad. No se detectaron patrones de colisión ni incidentes viales.")
        else:
            st.markdown(f"""
                <div style="background-color: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444; padding: 20px; border-radius: 12px; text-align: center;">
                    <h2 style="color: #ef4444; margin-top:0;">⚠️ REPORTE CRÍTICO DE ALERTA ⚠️</h2>
                    <p style="font-size: 18px; color: #f8fafc;">
                        Se detectaron un total de <b>{len(eventos_accidentes)} colisiones viales</b> en el video analizado.
                    </p>
                    <p style="color: #94a3b8; font-size: 15px;">
                        Los datos de marcas de tiempo han sido registrados para el reporte de incidentes del centro de control de emergencias.
                    </p>
                </div>
            """, unsafe_allow_html=True)
else:
    # Mensaje informativo inicial si no se ha cargado un video
    st.info("💡 Por favor, selecciona y carga un archivo de video en el cargador de arriba para comenzar el procesamiento.")
