# SIPID - Sistema de Inteligencia Predictiva para la Investigación Delictiva

SIPID es una plataforma de análisis geoespacial y aprendizaje automático orientada a la prevención e identificación delictiva. Procesa datos históricos de incidencia (2020-2025), genera inteligencia territorial con malla hexagonal H3 y visualiza resultados en un dashboard interactivo construido con Streamlit y Folium.

## Objetivo del Proyecto

Transformar registros históricos de delitos en predicciones accionables para identificar zonas de riesgo, priorizar vigilancia y apoyar la toma de decisiones operativas.

## Estructura del Proyecto

```text
SIPID/
|-- data/                   # Datos del proyecto (ignorado en Git)
|   |-- processed/          # Datos limpios y estructurados
|   `-- raw/                # Archivos originales (.xlsx 2020-2025)
|-- models/                 # Artefactos serializados (.joblib)
|   |-- grid_encoder.joblib
|   |-- modalidad_encoder.joblib
|   `-- sipid_best_model.joblib
|-- src/                    # Lógica de procesamiento y entrenamiento
|   |-- __init__.py
|   |-- process_data.py     # ETL y feature engineering
|   `-- train_model.py      # Benchmark y entrenamiento de modelos
|-- dashboard.py            # Interfaz forense (Streamlit)
|-- main.py                 # Orquestador del pipeline
|-- requirements.txt        # Dependencias del proyecto
`-- README.md
```

## Stack Tecnológico

- Python 3.10+
- Pandas, NumPy, Openpyxl
- H3, Geopy, Folium
- Scikit-learn, XGBoost, LightGBM
- Streamlit, Altair
- Joblib

## Pipeline de Datos (ETL)

1. Ingesta y unificación de archivos `.xlsx` históricos.
2. Limpieza de datos y normalización de campos.
3. Transformación temporal (`hora`, `día_semana`, `mes`).
4. Filtro espacial por bounding box para eliminar outliers.
5. Conversión de coordenadas a celdas H3 (resolución definida).
6. Codificación categórica para entrenamiento.

## Modelado Predictivo

SIPID evalúa un marco comparativo de ensambles para clasificación multiclase de riesgo por celda H3:

- Random Forest (baseline)
- XGBoost (challenger)
- LightGBM (challenger)

Métricas principales:

- F1-Score (Macro)
- Log-Loss
- Error geoespacial (Haversine)

El modelo con mejor desempeño global se serializa como `sipid_best_model.joblib`.

## Requisitos Previos

- Python 3.10 o superior
- `pip`
- Git (opcional para clonar)

## Instalación

```bash
git clone <URL_DEL_REPOSITORIO>
cd "SIPID - VGITHUB"
python -m venv venv
```

Activación del entorno virtual:

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

1. Coloca los archivos fuente en `data/raw/`.
2. Ejecuta el pipeline de preparación y entrenamiento:

```bash
python main.py
```

3. Inicia la interfaz web:

```bash
streamlit run dashboard.py
```

La aplicación abrirá en `http://localhost:8501`.

## Módulos del Dashboard

- Análisis Predictivo: estimación de probabilidad delictiva por tipo, fecha y horario.
- Análisis Temático: hotspots históricos por modalidad delictiva.
- Panorama Evolutivo: comportamiento territorial por año y tendencias.
- Estadística Forense: distribución por hora, día y zonas críticas.

