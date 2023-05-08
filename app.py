import streamlit as st  
from detector import * 
from streamlit import session_state as ss
import os 
import subprocess

if "process" not in ss:
    ss.process = None

def run_webcam():
    #global process
    ss.process= subprocess.Popen(['python', 'yolonas_webcam.py'])
    print("process started...")

def stop_webcam():
    #global process
    ss.process.terminate()
    # Wait for the process to terminate
    ss.process.wait()
    print("Process has been terminated.")


st.title("Detección de Objetos con YOLO NAS")
detector = Detector()


st.sidebar.title("Seleccione imágenes o videos")

file_options = st.sidebar.radio(label="**Tipo de detección**",options=["imagen","video","detección en tiempo real"])

if file_options =="imagen":
    imagen = st.sidebar.text_input(label="**Ingrese la url de la imagen**",
                                    placeholder="https://imagen_random.jpg")
    st.subheader("**Detección de objetos en imágenes**")
    if imagen:
        out_path = os.path.join("im_detections", "pred_0.jpg") 
        detector.onImage(imagen)
        #c1, c2 = st.columns(2)
        #with c1: 
        st.image(imagen, caption="Imagen original")
        #with c2:
        st.image(out_path, caption="Detección de objetos")


elif file_options =="video":
    st.subheader("**Detección de objetos en videos**")
    uploaded_video = st.sidebar.file_uploader("**Cargue un video (duración recomendada 10s)**", type=["mp4"])
    if uploaded_video:
        st.video(uploaded_video)
        vid_btt = st.button("Transformar video", type="primary")
        if vid_btt:
            os.remove("detection.mp4")
            with st.spinner("**Transformando el video...** (Esto puede tomar unos minutos)"):
                detector.onVideo(uploaded_video.name)

        if os.path.exists("detection.mp4"):
            video_file = open('detection.mp4', 'rb')
            video_bytes = video_file.read()
            st.download_button("Descargar video procesado", file_name="detection.mp4", data=video_bytes)


elif file_options == "detección en tiempo real":
    st.subheader("Detección en tiempo real")
    inicio_bttn = st.button("Iniciar", type="primary", on_click=run_webcam)
    fin_bttn = st.button("Finalizar", on_click=stop_webcam)










