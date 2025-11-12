# Proyecto: Predicción de Enfermedad Cardíaca (Heart Disease)

## Resumen

Este proyecto utiliza Machine Learning para predecir la presencia de enfermedades cardíacas a partir de datos clínicos y de estilo de vida. Se construyó un **Pipeline** que incluye limpieza de datos, ingeniería de características, selección de modelo y optimización, guardado como `mejor_pipeline_opt.joblib`.
El modelo final prioriza la métrica **F1-Score** importante para minimizar falsos negativos en problemas de salud.

## Fases del Proyecto
### 1. Limpieza y Preparación de Datos
- Se limpiaron datos faltantes y se codificaron variables categóricas.  
- Se generaron nuevas variables predictivas y se balancearon clases.  
- Se obtuvieron los datasets finales listos para modelado.

### 2. Modelado y Optimización
- Se probó un conjunto de modelos: Regresión Logística, XGBoost, Random Forest, Gradient Boosting, Decision Tree, AdaBoost y Naive Bayes.  
- Se construyó un **Pipeline de scikit-learn** para transformar datos y entrenar modelos.  
- Se seleccionó el mejor modelo con **Búsqueda Bayesiana**, guardado como `mejor_pipeline_opt.joblib`.

### 3. Despliegue
- Se creó una aplicación en **Streamlit** (`Despliegue.py`) que permite hacer predicciones interactivas usando el modelo final.  
- La app utiliza el pipeline para transformar los datos de entrada automáticamente y devolver probabilidades de enfermedad cardíaca.

## Requisitos
Instalar librerías necesarias desde `requirements.txt`:

```bash
pip install -r requirements.txt

## Cómo Ejecutar la Aplicación

1. Activar tu entorno virtual:

- **Windows:**
```bash
venv\Scripts\activate

-**Mac/Linux:**
source venv/bin/activate

Ejecutar la aplicación Streamlit:
streamlit run Despliegue.py

