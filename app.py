import streamlit as st  
from detector import * 
from streamlit import session_state as ss
import os 
import subprocess

# Initialize process to None in session state
if "process" not in ss:
    ss.process = None

# Function to start the webcam
def run_webcam():
    # Use session state to access the global process variable
    ss.process= subprocess.Popen(['python', 'yolonas_webcam.py'])
    st.write("Webcam process started...")
    #print("process started...")

# Function to stop the webcam
def stop_webcam():
    # Use session state to access the global process variable
    ss.process.terminate()
    # Wait for the process to terminate
    ss.process.wait()
    st.write("Webcam process has been terminated.")

# Set the title for the Streamlit app
st.title("Detección de Objetos con YOLO NAS")

# Create a Detector object
detector = Detector()

# Create the sidebar options for the user to choose from
st.sidebar.title("Seleccione imágenes o videos")
file_options = st.sidebar.radio(label="**Tipo de detección**",options=["imagen","video","detección en tiempo real"])


# If the user selects image detection
if file_options =="imagen":
    # Create a text input for the user to enter an image URL
    imagen = st.sidebar.text_input(label="**Ingrese la url de la imagen**",
                                    placeholder="https://imagen_random.jpg")
    # Show the subheader for object detection in images
    st.subheader("**Detección de objetos en imágenes**")
    # If the user has entered an image URL
    if imagen:
        # Define the path for the output image
        out_path = os.path.join("im_detections", "pred_0.jpg")
        # Call the onImage method of the Detector object to perform object detection on the image
        detector.onImage(imagen)
        # Display the original image and the image with object detections
        st.image(imagen, caption="Imagen original")
        st.image(out_path, caption="Detección de objetos")

# If the user selects video detection
elif file_options =="video":
    # Show the subheader for object detection in videos
    st.subheader("**Detección de objetos en videos**")
    # Create a file uploader for the user to upload a video
    uploaded_video = st.sidebar.file_uploader("**Cargue un video (duración recomendada 10s)**", type=["mp4"])
    # If the user has uploaded a video
    if uploaded_video:
        # Display the video
        st.video(uploaded_video)
        # Create a button to start processing the video
        vid_btt = st.button("Transformar video", type="primary")
        # If the user clicks the button
        if vid_btt:
            # Remove any previously processed videos
            os.remove("detection.mp4")
            # Show a spinner while the video is being processed
            with st.spinner("**Transformando el video...** (Esto puede tomar unos minutos)"):
                # Call the onVideo method of the Detector object to perform object detection on the video
                detector.onVideo(uploaded_video.name)
        # If the processed video exists
        if os.path.exists("detection.mp4"):
            # Open the processed video file
            video_file = open('detection.mp4', 'rb')
            # Read the video bytes
            video_bytes = video_file.read()
            # Create a download button for the processed video
            st.download_button("Descargar video procesado", file_name="detection.mp4", data=video_bytes)

# If the user selects real time object detection
elif file_options == "detección en tiempo real":
    # Show the subheader for real time object detection 
    st.subheader("Detección en tiempo real")
    #Creates a button that launchs the object detection when clicked
    inicio_bttn = st.button("Iniciar", type="primary", on_click=run_webcam)
    #Creates a button that ends the object detection when clicked
    fin_bttn = st.button("Finalizar", on_click=stop_webcam)










