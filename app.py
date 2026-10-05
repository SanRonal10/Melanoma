import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Detección de Melanoma", page_icon="🩺", layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write(
    "Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma."
)


# Cargar el modelo con TensorFlow/Keras o pickle de respaldo
@st.cache_resource
def load_model():
    try:
        # Intento 1: Carga nativa de Keras/TensorFlow
        return tf.keras.models.load_model("mimodelo.pkl")
    except Exception:
        import joblib

        # Intento 2: Carga directa por joblib sin envoltorios
        return joblib.load("mimodelo.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(
        f"Error al cargar el archivo 'mimodelo.pkl'. Asegúrate de que esté en la raíz de tu repositorio: {e}"
    )
    st.stop()


# Función para preprocesar la imagen
def preprocess_image(image):
    # Aseguramos formato RGB
    img = image.convert("RGB")
    # Redimensionamos a 224x224 (ajusta si entrenaste con otro tamaño)
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # Normalización (de 0 a 1)
    img_array = img_array / 255.0

    # Expandir dimensión para simular un lote: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


# Componente para subir la imagen
uploaded_file = st.file_uploader(
    "Carga una imagen (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Analizando la imagen..."):
            processed_img = preprocess_image(image)

            try:
                # Realizar predicción con la red neuronal
                predictions = model.predict(processed_img)

                # Extraer el valor de predicción
                prob = (
                    float(predictions[0][0])
                    if predictions.ndim > 1
                    else float(predictions[0])
                )

                if prob >= 0.5:
                    st.error(
                        f"⚠️ **Resultado:** Posible Melanoma detectado ({prob*100:.2f}% de probabilidad)."
                    )
                else:
                    st.success(
                        f"✅ **Resultado:** Posible Lesión Benigna ({(1-prob)*100:.2f}% de probabilidad)."
                    )
            except Exception as pred_err:
                st.error(f"Error al procesar la predicción: {pred_err}")
