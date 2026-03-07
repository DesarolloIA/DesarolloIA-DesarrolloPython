# 🗺️ SIPID - Sistema de Inteligencia Comercial Territorial

Este proyecto es una herramienta integral de analítica espacial y machine learning diseñada para procesar datos históricos (delitos/eventos), generar modelos predictivos de demanda basados en mallas hexagonales (H3) y visualizar los resultados a través de un dashboard interactivo utilizando Streamlit y Folium.

## 📂 Estructura del Proyecto

El repositorio sigue una arquitectura modular y organizada para separar los datos, los modelos y el código fuente:

```text
SIPID/
├── data/                   # Almacenamiento de datos (ignorado en Git por seguridad)
│   ├── processed/          # Datos limpios y procesados (ej. datos_limpios_para_modelo.csv)
│   └── raw/                # Archivos Excel originales (ej. DELITOS DE ALTO IMPACTO-202X.xlsx)
├── models/                 # Modelos entrenados y codificadores exportados (.joblib)
│   ├── grid_encoder.joblib
│   ├── modalidad_encoder.joblib
│   └── sipid_model.joblib
├── src/                    # Código fuente del pipeline de machine learning
│   ├── __init__.py
│   ├── process_data.py     # Script para limpieza, geofiltro y generación de la malla H3 (Fase 2)
│   └── train_model.py      # Script para entrenar el modelo Random Forest (Fase 3)
├── venv/                   # Entorno virtual de Python (local)
├── .gitignore              # Archivos y carpetas a ignorar por Git (venv, data, __pycache__, etc.)
├── dashboard.py            # Aplicación frontend en Streamlit (Interfaz gráfica y mapas)
├── main.py                 # Orquestador principal que ejecuta el procesamiento y entrenamiento
└── requirements.txt        # Dependencias y librerías necesarias del proyecto
```

## ⚙️ Requisitos Previos

Asegúrate de tener instalado en tu sistema:

- Python 3.8 o superior
- Git

## 🚀 Guía de Instalación

Sigue estos pasos para clonar e inicializar el proyecto en tu entorno local:

1. Clonar el repositorio:

```bash
git clone https://github.com/tu-usuario/tu-repositorio-sipid.git
cd tu-repositorio-sipid
```

2. Crear un Entorno Virtual: Es una buena práctica utilizar un entorno virtual para aislar las dependencias del proyecto.

```bash
python -m venv venv
```

3. Activar el Entorno Virtual:

En Windows:

```bash
venv\Scripts\activate
```

En macOS y Linux:

```bash
source venv/bin/activate
```

4. Instalar las dependencias: Con el entorno virtual activado, instala todas las librerías requeridas leídas desde tu archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 🛠️ Cómo Ejecutar el Proyecto

El flujo de trabajo se divide en dos grandes etapas: la preparación/entrenamiento del modelo y la visualización en el panel interactivo.

Paso 1: Preparar los Datos Originales

Asegúrate de colocar tus archivos de Excel originales con los datos geolocalizados dentro de la carpeta `data/raw/`. El sistema buscará archivos con extensión `.xlsx` y leerá la hoja correspondiente (por defecto `RoboAuto`).

Paso 2: Ejecutar el Pipeline de Procesamiento y Entrenamiento

Una vez que los datos estén en su lugar, ejecuta el script principal. Este script se encargará de limpiar los datos, generar los hexágonos H3, guardar el dataset procesado y entrenar el modelo predictivo (Random Forest):

```bash
python main.py
```

Si la ejecución es exitosa, verás en la consola los mensajes de "Datos guardados en..." y "Modelo guardado en..." confirmando que los archivos `.joblib` y el `.csv` están listos.

Paso 3: Levantar el Dashboard Interactivo

Para visualizar los mapas de calor, las predicciones de la Inteligencia Artificial y las estadísticas forenses espaciales, inicia el servidor de Streamlit:

```bash
streamlit run dashboard.py
```

Esto abrirá automáticamente una pestaña en tu navegador web predeterminado (usualmente en `http://localhost:8501`) mostrando el panel de control de SIPID.

## 📊 Módulos del Dashboard

Una vez dentro de la aplicación web, encontrarás tres pestañas principales:

- Análisis Predictivo: Permite seleccionar una línea de negocio, mes, día y ventana horaria para calcular las zonas de "Alta Oportunidad" usando el modelo Random Forest.
- Análisis Temático: Genera un mapa de calor histórico para analizar la densidad y concentración de transacciones o eventos pasados según su tipología.
- Panorama Territorial: Muestra la evolución global y consolidada de la actividad comercial dividida por años, utilizando una paleta de colores forense.
"# DesarrolloPython" 
