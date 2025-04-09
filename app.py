import streamlit as st
import joblib
import pandas as pd

# Cargar el modelo
model = joblib.load("best_model.pkl")

# Título de la app
st.title("Clasificación de Evaluación de Automóviles")

# Entradas del usuario
buying = st.selectbox("Precio de compra", ["low", "med", "high", "vhigh"])
maint = st.selectbox("Costo de mantenimiento", ["low", "med", "high", "vhigh"])
doors = st.selectbox("Número de puertas", ["2", "3", "4", "5more"])
persons = st.selectbox("Capacidad de personas", ["2", "4", "more"])
lug_boot = st.selectbox("Tamaño del maletero", ["small", "med", "big"])
safety = st.selectbox("Nivel de seguridad", ["low", "med", "high"])

# Crear DataFrame de entrada
input_data = pd.DataFrame({
    "buying": [buying],
    "maint": [maint],
    "doors": [doors],
    "persons": [persons],
    "lug_boot": [lug_boot],
    "safety": [safety]
})

# Predecir
if st.button("Predecir"):
    prediction = model.predict(input_data)[0]
    st.write(f"La evaluación del automóvil es: {prediction}")