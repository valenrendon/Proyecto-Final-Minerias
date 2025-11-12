import joblib, os

model_path = os.path.join(os.getcwd(), "mejor_pipeline_opt.joblib")
print("Cargando modelo desde:", model_path)

try:
    model = joblib.load(model_path)
    print(" Modelo cargado correctamente.")
    print("Tipo:", type(model))
except Exception as e:
    print(" Error al cargar el modelo:", e)
