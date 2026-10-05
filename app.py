import pickle
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


# Cargar el modelo guardado
@st.cache_resource
def load_model():
    with open("mimodelo.pkl", "rb") as file:
        model = pickle.load(file)
    return model


try:
    model = load_model()
except Exception as e:
    st.error(
        f"Error al cargar el archivo 'mimodelo.pkl'. Asegúrate de que esté en la raíz de tu repositorio: {e}"
    )
    st.stop()


# Función para preprocesar la imagen
def preprocess_image(image):
    # Ajusta el tamaño según las dimensiones de entrada que usaste al entrenar
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Si el modelo espera una entrada aplanada (1D) o normalizada:
    # img_array = img_array / 255.0  # Descomenta si normalizaste entradas
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
        with st.spinner("Procesando imagen..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)

            # Asume que 1 es Melanoma y 0 es Benigno (Ajusta según tu modelo)
            if prediction[0] == 1:
                st.error("⚠️ **Resultado:** Posible Melanoma detected.")
            else:
                st.success("✅ **Resultado:** Posible Lesión Benigna.")

            # Si el modelo soporta probabilidades (predict_proba)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(processed_img)
                st.info(f"Confianza de la predicción: {np.max(probs) * 100:.2f}%")
