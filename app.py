import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from google.cloud import storage
from river import linear_model, preprocessing, metrics

# =========================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea con River", page_icon="🌊")
st.title("🌊 Aprendizaje en línea con River (Streaming realista desde Cloud Storage)")

st.markdown("""
Este panel demuestra cómo un modelo de **aprendizaje incremental** puede entrenarse y actualizarse 
a partir de un dataset grande alojado en **Google Cloud Storage (GCS)**.  
Cada archivo CSV del bucket se procesa como un *fragmento temporal* del flujo de datos.
""")

# =========================================================
# FUNCIONES AUXILIARES PARA GUARDAR Y CARGAR EL MODELO
# =========================================================
def save_model_to_gcs(model, bucket_name, destination_blob):
    """Guarda el modelo en formato pickle dentro del bucket de GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado en GCS: `{destination_blob}`")
    except Exception as e:
        st.warning(f"No se pudo guardar el modelo: {e}")

def load_model_from_gcs(bucket_name, source_blob):
    """Carga el modelo desde GCS si existe, de lo contrario devuelve None."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        if blob.exists():
            data = blob.download_as_bytes()
            st.info("Modelo cargado desde GCS.")
            return pickle.loads(data)
        else:
            st.info("ℹ️ No se encontró un modelo previo, se iniciará uno nuevo.")
            return None
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo previo: {e}")
        return None

# =========================================================
# CONFIGURACIÓN DE PARÁMETROS
# =========================================================
bucket_name = st.text_input("Nombre del bucket de GCS:", "taxi-bucket-2022-jeko")
prefix = st.text_input("Carpeta/prefijo dentro del bucket:", "tlc_yellow_trips_2022/")
limite = st.number_input("Número de registros por archivo a procesar:", value=1000, step=100, min_value=100)
mostrar_grafico = st.checkbox("Mostrar gráfico de evolución del R²", value=True)

# =========================================================
# INICIALIZACIÓN DEL MODELO Y LAS MÉTRICAS
# =========================================================
MODEL_PATH = "models/model_incremental.pkl"

if "model" not in st.session_state:
    model = load_model_from_gcs(bucket_name, MODEL_PATH)
    if model is None:
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    st.session_state.model = model
    st.session_state.metric = metrics.R2()
    st.session_state.history = []  # Guarda evolución del R²

model = st.session_state.model
r2 = st.session_state.metric

# =========================================================
# FUNCIÓN DE STREAMING DESDE GCS (OPTIMIZADA Y CORREGIDA)
# =========================================================
def stream_from_bucket(bucket_name, prefix, limite=1000, chunksize=500):
    """
    Lee archivos CSV de GCS, entrena el modelo de forma incremental y produce el R2.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    st.info(f"Se encontraron {len(blobs)} archivos en `{prefix}`.")

    for idx, blob in enumerate(blobs, start=1):
        st.write(f"Procesando archivo {idx} de {len(blobs)}: `{blob.name.split('/')[-1]}`")

        try:
            content = blob.download_as_bytes()
            buffer = io.BytesIO(content)

            count = 0
            for chunk in pd.read_csv(buffer, chunksize=chunksize):
                if count >= limite:
                    break
                for _, row in chunk.iterrows():
                    if count >= limite:
                        break
                    try:
                        x = {
                            "dist": float(row["trip_distance"]),
                            "pass": float(row["passenger_count"]),
                            "hour": float(row.get("pickup_hour", 0))
                        }
                        y = float(row["fare_amount"])

                        # 1. Predecir (Test)
                        y_pred = model.predict_one(x)

                        # 2. Actualizar la métrica (Train)
                        #    FIX: Solo actualizar si la predicción es un número válido.
                        #    Esto evita el error de división por cero en la métrica R2
                        #    cuando el modelo aún no se ha "calentado".
                        if y_pred is not None and np.isfinite(y_pred):
                            r2.update(y, y_pred)

                        # 3. Aprender (Train)
                        model.learn_one(x, y)
                        
                        count += 1
                    except (ValueError, KeyError):
                        # Ser específico con las excepciones es una mejor práctica
                        continue
                if count >= limite:
                    break

        except Exception as e:
            st.warning(f"Error al procesar `{blob.name}`: {e}")
            continue

        yield blob.name, r2.get()

# =========================================================
# BOTÓN DE ACTUALIZACIÓN DEL MODELO
# =========================================================
if st.button("Actualizar modelo con datos del bucket"):
    st.info("Procesando archivos desde el bucket... esto puede tardar unos minutos ⏳")

    progreso = st.progress(0)
    valores = []

    # Se necesita un cliente de storage para listar los blobs
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    total = len(blobs)

    if total > 0:
        for i, (fname, score) in enumerate(stream_from_bucket(bucket_name, prefix, limite)):
            # Si el score es None o NaN, no lo agregamos a la gráfica para evitar errores
            if score is not None and np.isfinite(score):
                 valores.append(score)
                 st.session_state.history.append(score)

            progreso.progress(min((i + 1) / total, 1.0))
            st.write(f"{fname.split('/')[-1]} — R² acumulado: **{r2.get():.4f}**")

        progreso.empty()
        st.success("¡Entrenamiento incremental completado!")

        # Guardar modelo actualizado en GCS
        save_model_to_gcs(model, bucket_name, MODEL_PATH)

        if mostrar_grafico and valores:
            st.line_chart(
                pd.DataFrame({"R²": valores}, index=np.arange(1, len(valores) + 1)),
                height=300,
                use_container_width=True
            )
    else:
        progreso.empty()
        st.warning("No se encontraron archivos en el bucket/prefijo especificado.")

# =========================================================
# SECCIÓN FINAL: ESTADO ACTUAL DEL MODELO
# =========================================================
st.markdown("---")
st.subheader("Estado actual del modelo")
st.write(f"R² actual: **{r2.get():.4f}**")

if st.session_state.history:
    st.line_chart(st.session_state.history, height=200, use_container_width=True)

st.caption("Cloud Run + River • Dataset público de taxis NYC (2022)")
