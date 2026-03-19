from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predicción de depósito bancario", layout="centered")

MODEL_PATH = Path("modelo_final.joblib")
DATA_PATH = Path("Datos/bank_09.pkl")


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


@st.cache_data
def load_reference_data(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def unique_in_order(series: pd.Series) -> list:
    return series.dropna().astype(str).drop_duplicates().tolist()


def etiqueta_columna(col: str) -> str:
    etiquetas = {
        "age": "Edad",
        "job": "Tipo de trabajo",
        "marital": "Estado civil",
        "education": "Nivel de educación",
        "default": "¿Tiene créditos impagados?",
        "balance": "Balance anual medio",
        "housing": "¿Tiene hipoteca?",
        "loan": "¿Tiene préstamo personal?",
        "contact": "Tipo de contacto",
        "day": "Último día de contacto",
        "month": "Último mes de contacto",
        "duration": "Duración del último contacto (segundos)",
        "campaign": "Número de contactos en esta campaña",
        "previous": "Número de contactos anteriores",
        "poutcome": "Resultado de campañas anteriores",
        "contactado_antes": "¿Ha sido contactado antes?",
        "pdays": "Días desde el contacto anterior",
    }
    return etiquetas.get(col, col)


if not MODEL_PATH.exists():
    st.error(f"No se encuentra el modelo final en: {MODEL_PATH}")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"No se encuentra el dataset de referencia en: {DATA_PATH}")
    st.stop()

modelo_final = load_model(MODEL_PATH)
df_ref = load_reference_data(DATA_PATH).copy()

if "deposit" in df_ref.columns:
    df_ref_sin_target = df_ref.drop(columns=["deposit"]).copy()
else:
    df_ref_sin_target = df_ref.copy()

if hasattr(modelo_final, "feature_names_in_"):
    columnas_esperadas = list(modelo_final.feature_names_in_)
else:
    columnas_esperadas = df_ref_sin_target.columns.tolist()

# Dataset de referencia alineado con el modelo para extraer opciones y medianas
# Si el modelo espera 'contactado_antes', reproducimos el mismo paso que en entrenamiento.
df_ref_modelo = df_ref_sin_target.copy()
if "contactado_antes" in columnas_esperadas and "pdays" in df_ref_modelo.columns:
    df_ref_modelo["contactado_antes"] = (df_ref_modelo["pdays"] != -1).astype(int)
    df_ref_modelo = df_ref_modelo.drop(columns=["pdays"])

for col in columnas_esperadas:
    if col not in df_ref_modelo.columns:
        df_ref_modelo[col] = np.nan

df_ref_modelo = df_ref_modelo.reindex(columns=columnas_esperadas)

categoricas = [
    col for col in columnas_esperadas
    if col in df_ref_modelo.columns and str(df_ref_modelo[col].dtype) in ["object", "category", "bool"]
]

opciones = {
    col: unique_in_order(df_ref_modelo[col])
    for col in categoricas
}

medianas = {}
for col in columnas_esperadas:
    if col in df_ref_modelo.columns and pd.api.types.is_numeric_dtype(df_ref_modelo[col]):
        serie = df_ref_modelo[col].dropna()
        medianas[col] = float(serie.median()) if not serie.empty else 0.0

# Casos de prueba para el PDF
casos_prueba = {
    "Manual": None,
    "Caso 1": {
        "age": 39,
        "job": "unemployed",
        "marital": "divorced",
        "education": "unknown",
        "default": "yes",
        "balance": 549.50,
        "housing": "yes",
        "loan": "yes",
        "contact": "cellular",
        "month": "dec",
        "duration": 255,
        "campaign": 1,
        "previous": 2,
        "poutcome": "unknown",
        "pdays": 3,
        "contactado_antes": 1,
        "day": 15,
    },
    "Caso 2": {
        "age": 52,
        "job": "technician",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1200.00,
        "housing": "no",
        "loan": "no",
        "contact": "telephone",
        "month": "jul",
        "duration": 180,
        "campaign": 2,
        "previous": 0,
        "poutcome": "unknown",
        "pdays": -1,
        "contactado_antes": 0,
        "day": 7,
    },
}

st.title("Predicción de subscripción a depósito")
st.write(
    "Introduce los datos de un cliente nuevo o selecciona uno de los casos de prueba. "
    "La aplicación usa la pipeline final guardada en `modelo_final.joblib` para predecir "
    "si contrataría o no un depósito."
)

caso_seleccionado = st.selectbox(
    "Selecciona un caso de prueba para autocompletar el formulario",
    list(casos_prueba.keys())
)
preset = casos_prueba[caso_seleccionado]

if preset is not None:
    st.info(
        f"Se ha cargado automáticamente {caso_seleccionado}. Ahora puedes pulsar directamente en 'Predecir' o ajustar algún valor."
    )

with st.form("formulario_cliente"):
    st.subheader("Datos del cliente")
    col1, col2 = st.columns(2)

    valores = {}

    for i, col in enumerate(columnas_esperadas):
        contenedor = col1 if i % 2 == 0 else col2
        label = etiqueta_columna(col)
        valor_preset = preset.get(col) if preset is not None else None

        with contenedor:
            if col == "contactado_antes":
                indice_default = 1 if valor_preset == 1 else 0
                opcion = st.selectbox(label, ["no", "sí"], index=indice_default)
                valores[col] = 1 if opcion == "sí" else 0
            elif col in opciones and len(opciones[col]) > 0:
                lista_opciones = opciones[col]
                if valor_preset in lista_opciones:
                    indice_default = lista_opciones.index(valor_preset)
                else:
                    indice_default = 0
                valores[col] = st.selectbox(label, lista_opciones, index=indice_default)
            elif col == "day":
                valores[col] = int(
                    st.number_input(
                        label,
                        min_value=1,
                        max_value=31,
                        value=int(valor_preset) if valor_preset is not None else int(round(medianas.get(col, 15))),
                        step=1,
                    )
                )
            elif col == "age":
                valores[col] = int(
                    st.number_input(
                        label,
                        min_value=18,
                        max_value=100,
                        value=int(valor_preset) if valor_preset is not None else int(round(medianas.get(col, 40))),
                        step=1,
                    )
                )
            elif col in ["duration", "campaign", "previous"]:
                valor_defecto = int(valor_preset) if valor_preset is not None else max(0, int(round(medianas.get(col, 0))))
                valores[col] = int(
                    st.number_input(label, min_value=0, value=valor_defecto, step=1)
                )
            elif col == "balance":
                valor_defecto = float(valor_preset) if valor_preset is not None else float(medianas.get(col, 0.0))
                valores[col] = float(
                    st.number_input(label, value=valor_defecto, step=100.0)
                )
            else:
                valor_defecto = float(valor_preset) if valor_preset is not None else float(medianas.get(col, 0.0))
                valores[col] = float(
                    st.number_input(label, value=valor_defecto, step=1.0)
                )

    enviado = st.form_submit_button("Predecir")

if enviado:
    X_nuevo = pd.DataFrame([valores]).reindex(columns=columnas_esperadas)

    columnas_nan = X_nuevo.columns[X_nuevo.isna().any()].tolist()
    if columnas_nan:
        st.error(
            "No se puede predecir porque faltan valores en estas variables: "
            + ", ".join(columnas_nan)
            + ". Revisa que el formulario cubra exactamente las columnas del modelo."
        )
        st.subheader("Entrada utilizada por la pipeline")
        st.dataframe(X_nuevo, use_container_width=True)
        st.stop()

    pred = modelo_final.predict(X_nuevo)[0]
    etiqueta = "yes" if pred in [1, "yes", True] else "no"

    st.subheader("Resultado de la predicción")
    if etiqueta == "yes":
        st.success("Predicción: yes. El modelo estima que el cliente sí contrataría el depósito.")
    else:
        st.error("Predicción: no. El modelo estima que el cliente no contrataría el depósito.")

    if hasattr(modelo_final, "predict_proba"):
        try:
            proba = modelo_final.predict_proba(X_nuevo)[0]
            clases = list(modelo_final.classes_)
            if 1 in clases:
                idx_yes = clases.index(1)
            elif "yes" in clases:
                idx_yes = clases.index("yes")
            else:
                idx_yes = int(np.argmax(proba))
            st.write(f"Probabilidad estimada de 'yes': {proba[idx_yes]:.4f}")
        except Exception:
            pass

    st.subheader("Entrada utilizada por la pipeline")
    st.dataframe(X_nuevo, use_container_width=True)

st.caption(
    "La aplicación construye el formulario a partir de las columnas que espera el modelo final y permite "
    "autocompletar dos casos de prueba para facilitar la comprobación pedida en el PDF."
)
