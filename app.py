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


# Cargar el modelo guardado utilizando joblib
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


# Función para preprocesar la imagen ingresada
def preprocess_image(image):
    # Redimensiona según las dimensiones con las que entrenaste tu modelo (ej. 224x224)
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Aplanar la imagen para modelos de Scikit-Learn (1D array)
    img_flat = img_array.flatten().reshape(1, -1)

    return img_flat


# Carga de la imagen por el usuario
uploaded_file = st.file_uploader(
    "Carga una imagen (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Procesando imagen con el modelo..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)

            # Clasificación de la respuesta (Ajusta la lógica si 0 o 1 corresponden a otra etiqueta)
            if prediction[0] == 1:
                st.error("⚠️ **Resultado:** Posible Melanoma detectado.")
            else:
                st.success("✅ **Resultado:** Posible Lesión Benigna.")

            # Mostrar probabilidad si el modelo soporta predict_proba
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(processed_img)
                confianza = np.max(probs) * 100
                st.info(f"Nivel de confianza de la predicción: {confianza:.2f}%")
