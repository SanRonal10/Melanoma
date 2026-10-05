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


# Carga con pickle directo en modo binario "rb" para evitar el bug de joblib/_io.BytesIO
@st.cache_resource
def load_model():
    # Intentamos primero con dill (si el modelo incluye lambdas/custom layers)
    try:
        import dill

        with open("mimodelo.pkl", "rb") as f:
            return dill.load(f)
    except Exception:
        # Respaldo con pickle estándar
        with open("mimodelo.pkl", "rb") as f:
            return pickle.load(f)


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
    # Redimensionar al tamaño estándar de entrada
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # Escalado/normalización a [0, 1]
    img_array = img_array / 255.0

    # Expandir dimensión para simular el lote (batch): (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, img_array.flatten().reshape(1, -1)


# Cargar imagen
uploaded_file = st.file_uploader(
    "Carga una imagen (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Procesando imagen..."):
            img_batch, img_flat = preprocess_image(image)

            # Intentar predicción según la estructura del modelo
            try:
                # Caso 1: Redes Neuronales / Keras (Esperan tensor 4D)
                prediction = model.predict(img_batch)
            except Exception:
                # Caso 2: Scikit-learn / XGBoost (Esperan vector aplanado 2D)
                prediction = model.predict(img_flat)

            # Extraer valor escalar de la predicción
            if isinstance(prediction, (list, np.ndarray)):
                pred_val = prediction[0]
                if isinstance(pred_val, (list, np.ndarray)):
                    pred_val = pred_val[0]
            else:
                pred_val = prediction

            # Mostrar resultado
            if float(pred_val) >= 0.5 or pred_val == 1:
                st.error("⚠️ **Resultado:** Posible Melanoma detectado.")
            else:
                st.success("✅ **Resultado:** Posible Lesión Benigna.")
