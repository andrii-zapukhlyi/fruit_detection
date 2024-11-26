import streamlit as st
from ultralytics import YOLO
#import cv2

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    model = YOLO("model/trained_YOLO/weights/best.pt")
    return model

model = load_model()

def make_prediction(image):
    prediction = model.predict(image, imgsz=640, verbose=False)
    plotted_img = prediction[0].plot()
    img_rgb = plotted_img#cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
    names = model.names
    classes_indices = prediction[0].boxes.cls.cpu().numpy()
    classes = [names[int(cls_idx)] for cls_idx in classes_indices]
    classes = list(set(classes))
    return img_rgb, classes


col1, col2 = st.columns([1,2])

with col1:
    st.title("Fruit Classifier")
    st.write("Upload an image or take a photo to classify fruits.")
    mode = st.radio("Choose mode:", ["Upload Photo", "Take Photo"], index=0)

with col2:
    if mode == "Upload Photo" or mode == "Take Photo":
        if mode == "Upload Photo":
            image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
        elif mode == "Take Photo":
            image = st.camera_input("Capture a photo")

        if image is not None:
            with open("temp_image.jpg", "wb") as f:
                f.write(image.getbuffer())
            predicted_image, prediction_classes = make_prediction("temp_image.jpg")
            st.image(predicted_image, caption="Prediction Result", use_column_width=True)
            st.write("Detected fruits:")
            if len(prediction_classes) == 0:
                st.write("No fruits detected")
            else:
                st.write(", ".join(prediction_classes))
        else:
            st.write("Please provide an image to predict.")
