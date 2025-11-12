import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# CONFIGURACIÓN INICIAL
st.set_page_config(page_title="Predicción de Enfermedad Cardíaca", page_icon="❤️", layout="centered")


# CARGA DEL MODELO
@st.cache_resource
def load_model():
    model_path = r"C:\Users\Valentina Rendón\Downloads\Universidad\Proyecto minerias\mejor_pipeline_opt.joblib"
    st.write(f"Cargando modelo desde: {model_path}")

    if not os.path.exists(model_path):
        st.error(f"No se encontró el archivo del modelo en: {model_path}")
        st.stop()

    try:
        model = joblib.load(model_path)
        st.success("Modelo cargado correctamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.info("Verifica que la versión de scikit-learn sea ≈ 1.6.0 (igual a la usada en Colab).")
        st.stop()

model = load_model()


# PRUEBA RÁPIDA DEL MODELO

st.write("### Prueba rápida del modelo:")
try:
    dummy = pd.DataFrame([{
        "BMI": 25,
        "Smoking": "No",
        "AlcoholDrinking": "No",
        "Stroke": "No",
        "PhysicalHealth": 5,
        "MentalHealth": 5,
        "DiffWalking": "No",
        "Sex": "Female",
        "AgeCategory": "55-59",
        "Race": "White",
        "Diabetic": "No",
        "PhysicalActivity": "Yes",
        "GenHealth": "Good",
        "SleepTime": 7,
        "Asthma": "No",
        "KidneyDisease": "No",
        "SkinCancer": "No"
    }])
    base_prob = model.predict_proba(dummy)[0][1]
    st.success(f"Probabilidad base de enfermedad cardíaca: {base_prob:.3f}")
except Exception as e:
    st.error(f"Error en la prueba del modelo: {e}")


# INTERFAZ VISUAL
st.title("Predicción de Enfermedad Cardíaca")
st.markdown(
    "El propósito de esta fase es implementar el modelo final entrenado en un entorno accesible "
    "para el usuario final, permitiendo realizar predicciones de manera interactiva mediante una interfaz desarrollada con **Streamlit**."
)

st.subheader("Ingrese los datos del paciente:")

Smoking = "Yes" if st.selectbox("¿Ha fumado alguna vez?", ["Sí", "No"]) == "Sí" else "No"
AlcoholDrinking = "Yes" if st.selectbox("¿Consume alcohol en exceso?", ["Sí", "No"]) == "Sí" else "No"
Stroke = "Yes" if st.selectbox("¿Ha tenido un accidente cerebrovascular?", ["Sí", "No"]) == "Sí" else "No"
DiffWalking = "Yes" if st.selectbox("¿Dificultad para caminar o subir escaleras?", ["Sí", "No"]) == "Sí" else "No"
Sex = "Male" if st.selectbox("Sexo", ["Masculino", "Femenino"]) == "Masculino" else "Female"

AgeCategory_map = {
    "18-24": "18-24", "25-29": "25-29", "30-34": "30-34", "35-39": "35-39",
    "40-44": "40-44", "45-49": "45-49", "50-54": "50-54", "55-59": "55-59",
    "60-64": "60-64", "65-69": "65-69", "70-74": "70-74", "75-79": "75-79",
    "80 o más": "80 or older"
}
AgeCategory = AgeCategory_map[st.selectbox("Grupo de edad", list(AgeCategory_map.keys()))]

Race_map = {
    "Blanco": "White", "Negro": "Black", "Asiático": "Asian",
    "Indígena americano/Alaska Nativo": "American Indian/Alaskan Native",
    "Otro": "Other", "Hispano": "Hispanic"
}
Race = Race_map[st.selectbox("Raza / Grupo étnico", list(Race_map.keys()))]

Diabetic_map = {
    "Sí": "Yes", "No": "No", "No, diabetes límite": "No, borderline diabetes",
    "Sí (durante el embarazo)": "Yes (during pregnancy)"
}
Diabetic = Diabetic_map[st.selectbox("Diagnóstico de diabetes", list(Diabetic_map.keys()))]

PhysicalActivity = "Yes" if st.selectbox("¿Realiza actividad física en los últimos 30 días?", ["Sí", "No"]) == "Sí" else "No"

GenHealth_map = {
    "Excelente": "Excellent", "Muy buena": "Very good", "Buena": "Good",
    "Regular": "Fair", "Mala": "Poor"
}
GenHealth = GenHealth_map[st.selectbox("Salud general percibida", list(GenHealth_map.keys()))]

Asthma = "Yes" if st.selectbox("¿Ha sido diagnosticado con asma?", ["Sí", "No"]) == "Sí" else "No"
KidneyDisease = "Yes" if st.selectbox("¿Tiene enfermedad renal?", ["Sí", "No"]) == "Sí" else "No"
SkinCancer = "Yes" if st.selectbox("¿Tiene cáncer de piel?", ["Sí", "No"]) == "Sí" else "No"

BMI = st.number_input("Índice de masa corporal (BMI)", min_value=10.0, max_value=60.0, value=25.0)
PhysicalHealth = st.slider("Días con mala salud física (últimos 30 días)", 0, 30, 0)
MentalHealth = st.slider("Días con mala salud mental (últimos 30 días)", 0, 30, 0)
SleepTime = st.slider("Horas promedio de sueño en 24 horas", 0, 24, 7)

input_data = pd.DataFrame([{
    "BMI": BMI,
    "Smoking": Smoking,
    "AlcoholDrinking": AlcoholDrinking,
    "Stroke": Stroke,
    "PhysicalHealth": PhysicalHealth,
    "MentalHealth": MentalHealth,
    "DiffWalking": DiffWalking,
    "Sex": Sex,
    "AgeCategory": AgeCategory,
    "Race": Race,
    "Diabetic": Diabetic,
    "PhysicalActivity": PhysicalActivity,
    "GenHealth": GenHealth,
    "SleepTime": SleepTime,
    "Asthma": Asthma,
    "KidneyDisease": KidneyDisease,
    "SkinCancer": SkinCancer
}])

st.write("### Datos ingresados:")
st.dataframe(input_data)


# PREDICCIÓN Y CALIBRACIÓN DE RIESGO

if st.button("Predecir"):
    try:
        prob = model.predict_proba(input_data)[0][1]
        umbral = 0.4  # Ajustable si el modelo subestima positivos
        pred = 1 if prob >= umbral else 0

        st.subheader("Resultado de la predicción:")
        if pred == 1:
            st.error(f"💔 El modelo predice **enfermedad cardíaca** (Prob: {prob:.2f})")
        else:
            st.success(f"❤️ El modelo predice **sin enfermedad cardíaca** (Prob: {prob:.2f})")

        st.caption(f"Umbral de decisión: {umbral:.2f} — puedes ajustarlo para controlar la sensibilidad del modelo.")

    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        st.write("Verifica que las categorías de entrada coincidan con las del modelo entrenado.")

# ANÁLISIS DE RIESGO (modo prueba)

st.sidebar.header("Análisis del modelo (solo pruebas)")

if st.sidebar.checkbox("Mostrar combinaciones con alto riesgo"):
    import itertools
    import numpy as np

    st.subheader("Casos con mayor probabilidad de enfermedad cardíaca")
    st.info("Este bloque genera combinaciones simuladas para identificar los perfiles más riesgosos según el modelo.")

    # Factores clave de riesgo
    riesgos = {
        "Smoking": ["Yes", "No"],
        "Stroke": ["Yes", "No"],
        "Diabetic": ["Yes", "No"],
        "DiffWalking": ["Yes", "No"],
        "GenHealth": ["Excellent", "Good", "Fair", "Poor"],
        "KidneyDisease": ["Yes", "No"]
    }

    # Valores base
    base = {
        "BMI": 28,
        "PhysicalHealth": 10,
        "MentalHealth": 5,
        "SleepTime": 6,
        "AgeCategory": "65-69",
        "Race": "White",
        "AlcoholDrinking": "No",
        "Sex": "Male",
        "PhysicalActivity": "No",
        "Asthma": "No",
        "SkinCancer": "No"
    }

    # Generar combinaciones
    comb_riesgos = list(itertools.product(
        riesgos["Smoking"], riesgos["Stroke"], riesgos["Diabetic"],
        riesgos["DiffWalking"], riesgos["GenHealth"], riesgos["KidneyDisease"]
    ))

    test_df = pd.DataFrame([
        {
            **base,
            "Smoking": s,
            "Stroke": st,
            "Diabetic": d,
            "DiffWalking": w,
            "GenHealth": g,
            "KidneyDisease": k
        }
        for (s, st, d, w, g, k) in comb_riesgos
    ])

    # Calcular probabilidades
    test_df["Prob_HeartDisease"] = model.predict_proba(test_df)[:, 1]
    test_df_sorted = test_df.sort_values(by="Prob_HeartDisease", ascending=False)

    st.write("### 🔝 Casos con mayor riesgo:")
    st.dataframe(test_df_sorted.head(10))

    max_prob = test_df_sorted["Prob_HeartDisease"].max()
    st.success(f"Máxima probabilidad encontrada: {max_prob:.3f}")

    if max_prob < 0.5:
        st.warning("El modelo puede estar subestimando los casos de riesgo. Considera revisar el balance de clases o el umbral de decisión.")