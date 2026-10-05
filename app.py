import io
import pickle
import zlib
import joblib
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Detección de Melanoma", page_icon="🩺", layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write(
    "Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma."
)


# Carga con descompresión inteligente de cabecera
@st.cache_resource
def load_model():
    # Intento 1: Joblib nativo directo desde la ruta del archivo
    try:
        return joblib.load("mimodelo.pkl")
    except Exception:
        pass

    # Intento 2: Lectura binaria y descompresión si la clave inicia con 'x' (zlib)
    with open("mimodelo.pkl", "rb") as f:
        content = f.read()

    if content.startswith(b"x"):
        decompressed = zlib.decompress(content)
        return pickle.loads(decompressed)

    return pickle.loads(content)


try:
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar el archivo 'mimodelo.pkl': {e}")
    st.stop()


# Preprocesamiento de la imagen
def preprocess_image(image):
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # Matriz 2D aplanada para Scikit-Learn / XGBoost
    flat_features = img_array.flatten().reshape(1, -1)

    # Tensor 4D normalizado para TensorFlow / Keras
    tensor_features = np.expand_dims(img_array / 255.0, axis=0)

    return flat_features, tensor_features


uploaded_file = st.file_uploader(
    "Carga una imagen (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Procesando imagen..."):
            flat_img, tensor_img = preprocess_image(image)

            try:
                prediction = model.predict(flat_img)
            except Exception:
                prediction = model.predict(tensor_img)

            if isinstance(prediction, (list, np.ndarray)):
                pred_val = prediction[0]
                if isinstance(pred_val, (list, np.ndarray)):
                    pred_val = pred_val[0]
            else:
                pred_val = prediction

            pred_num = float(pred_val)

            if pred_num >= 0.5 or pred_num == 1:
                st.error("⚠️ **Resultado:** Posible Melanoma detectado.")
            else:
                st.success("✅ **Resultado:** Posible Lesión Benigna.")
