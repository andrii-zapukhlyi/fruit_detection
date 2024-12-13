import streamlit as st
from ultralytics import YOLO
import cv2

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    model = YOLO("model/trained_YOLO/weights/best.pt")
    return model

model = load_model()

def make_prediction(image):
    prediction = model.predict(image, imgsz=640, verbose=False)
    plotted_img = prediction[0].plot()
    img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
    names = model.names
    classes_indices = prediction[0].boxes.cls.cpu().numpy()
    classes = [names[int(cls_idx)] for cls_idx in classes_indices]
    classes = list(set(classes))
    return img_rgb, classes


col1, col2 = st.columns([1,2])

with col1:
    st.title("Fruit Classifier")
    st.write("Upload an image, take a photo, or use live camera feed to classify fruits.")
    mode = st.radio("Choose mode:", ["Upload Photo", "Take Photo", "Live Camera (only for local app running)"], index=0)
    st.subheader("Fruits the model can classify:")
    fruits = [
        "banana-bag", "banana", "blackberries", "raspberry",
        "lemon-bag", "lemon", "grapes-bag", "grapes",
        "tomato-bag", "tomato", "apple-bag", "apple",
        "chili-bag", "chili"
    ]
    st.write("\n".join([f"- {fruit}" for fruit in fruits]))

with col2:
    if mode == "Live Camera (only for local app running)":
        st.write("Live camera working only when running the app locally.")
        run_camera = st.checkbox("Start Live Camera", key="start_live_camera")
        
        if run_camera:
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            prediction_text = st.empty()

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Unable to access the camera. Maybe you're trying to access the camera from app deployed on a Streamlit cloud.")
                    break

                predicted_frame, prediction_classes = make_prediction(frame)

                frame_placeholder.image(predicted_frame, channels="RGB", use_column_width=True)

                if len(prediction_classes) == 0:
                    prediction_text.write("No fruits detected")
                else:
                    prediction_text.write(f"Live fruits detected: {', '.join(prediction_classes)}")

            cap.release()
            cv2.destroyAllWindows()

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
