import joblib
import numpy as np
import streamlit as st
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Detección de Melanoma", page_icon="🩺", layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write(
    "Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma."
)


# Cargar modelo utilizando joblib/pickle
@st.cache_resource
def load_model():
    return joblib.load("mimodelo.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(
        f"Error al cargar el archivo 'mimodelo.pkl'. Asegúrate de que esté en la raíz de tu repositorio: {e}"
    )
    st.stop()


# Preprocesamiento de la imagen para Redes Neuronales Keras
def preprocess_image(image):
    # Aseguramos formato RGB
    img = image.convert("RGB")
    # Redimensionamos al tamaño típico de entrada de la red (ej. 224x224 o 150x150 según tu entrenamiento)
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # Normalización si entrenaste escalando píxeles de 0 a 1
    img_array = img_array / 255.0

    # Expandimos dimensión para simular el lote/batch: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


# Componente para subir imagen
uploaded_file = st.file_uploader(
    "Carga una imagen (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Analizando la imagen con la red neuronal..."):
            processed_img = preprocess_image(image)
            raw_pred = model.predict(processed_img)

            # Si el modelo devuelve una probabilidad sigmoide/softmax
            prob = float(raw_pred[0][0]) if raw_pred.ndim > 1 else float(raw_pred[0])

            # Umbral estándar a 0.5 (Ajusta la lógica si la clase 1 o 0 representa otra etiqueta)
            if prob >= 0.5:
                st.error(f"⚠️ **Resultado:** Posible Melanoma detectado ({prob*100:.2f}% probabilidad).")
            else:
                st.success(f"✅ **Resultado:** Posible Lesión Benigna ({(1-prob)*100:.2f}% probabilidad).")
