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


# Carga segura del modelo pasando únicamente la ruta como string
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


# Preprocesamiento de la imagen ingresada
def preprocess_image(image):
    # Convertir a formato RGB
    img = image.convert("RGB")

    # Redimensionar a la resolución usada durante el entrenamiento (ej. 224x224)
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # Si tu modelo requiere un vector 1D (Scikit-Learn) o un tensor 4D (Keras/TensorFlow)
    if hasattr(model, "predict"):
        # Intentamos primero la forma estándar para Keras/TensorFlow (1, 224, 224, 3)
        # Si usaste normalización de 0 a 1: img_array = img_array / 255.0
        try:
            return np.expand_dims(img_array, axis=0)
        except Exception:
            # Si falla, aplanamos para modelos tradicionales (1, N_features)
            return img_array.flatten().reshape(1, -1)

    return img_array


# Interfaz para subida de imagen
uploaded_file = st.file_uploader(
    "Carga una imagen (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Procesando imagen..."):
            processed_img = preprocess_image(image)

            try:
                raw_pred = model.predict(processed_img)

                # Extraer el valor numérico de la predicción
                if isinstance(raw_pred, (list, np.ndarray)):
                    pred_val = raw_pred[0]
                    if isinstance(pred_val, (list, np.ndarray)):
                        pred_val = pred_val[0]
                else:
                    pred_val = raw_pred

                # Evaluar según umbral
                if pred_val >= 0.5 or pred_val == 1:
                    st.error("⚠️ **Resultado:** Posible Melanoma detectado.")
                else:
                    st.success("✅ **Resultado:** Posible Lesión Benigna.")

            except Exception as err_pred:
                # Si el modelo espera 1D aplanado en lugar de 4D
                processed_img_flat = np.array(image.convert("RGB").resize((224, 224))).flatten().reshape(1, -1)
                raw_pred = model.predict(processed_img_flat)
                pred_val = raw_pred[0]

                if pred_val >= 0.5 or pred_val == 1:
                    st.error("⚠️ **Resultado:** Posible Melanoma detectado.")
                else:
                    st.success("✅ **Resultado:** Posible Lesión Benigna.")
