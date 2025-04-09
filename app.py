import streamlit as st
import joblib
import pandas as pd

# Título de la aplicación
st.title("Clasificación de Evaluación de Automóviles")

# Explicación de las etiquetas
st.sidebar.markdown(
    """
    ### Significado de las etiquetas:
    - **unacc**: Inaceptable
    - **acc**: Aceptable
    - **good**: Bueno
    - **vgood**: Muy bueno
    """
)

# Cargar el modelo entrenado (pipeline completo)
model = joblib.load("best_model.pkl")

# Cargar el LabelEncoder usado durante el entrenamiento
label_encoder = joblib.load("label_encoder.pkl")  # Asegúrate de haber guardado el LabelEncoder

# Entradas del usuario
st.header("Ingrese los detalles del automóvil:")
buying = st.selectbox("Precio de compra", ["low", "med", "high", "vhigh"])
maint = st.selectbox("Costo de mantenimiento", ["low", "med", "high", "vhigh"])
doors = st.selectbox("Número de puertas", ["2", "3", "4", "5more"])
persons = st.selectbox("Capacidad de personas", ["2", "4", "more"])
lug_boot = st.selectbox("Tamaño del maletero", ["small", "med", "big"])
safety = st.selectbox("Nivel de seguridad", ["low", "med", "high"])

# Crear un DataFrame con los datos ingresados
input_data = pd.DataFrame({
    "buying": [buying],
    "maint": [maint],
    "doors": [doors],
    "persons": [persons],
    "lug_boot": [lug_boot],
    "safety": [safety]
})

# Botón para realizar la predicción
if st.button("Predecir"):
    try:
        # Realizar la predicción usando el pipeline
        prediction = model.predict(input_data)[0]
        
        # Decodificar la predicción a la etiqueta original
        decoded_prediction = label_encoder.inverse_transform([prediction])[0]
        
        # Mostrar el resultado
        st.subheader("Resultado de la evaluación:")
        st.write(f"La evaluación del automóvil es: **{decoded_prediction}**")
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
