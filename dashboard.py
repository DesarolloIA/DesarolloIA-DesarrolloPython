# -*- coding: utf-8 -*-
# dashboard.py
# VERSIÓN V2: HEATMAP COLOR SCALE (YELLOW -> ORANGE -> RED)
# Ejecutar con: python -m streamlit run dashboard.py

import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import os
import altair as alt
import h3
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="SIPID | Intelligence Hub V2",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ESTILOS CSS (POLISHED & CLEAN) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* BASE */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1e293b;
    }

    .stApp { background-color: #f8fafc; }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #f1f5f9;
    }

    /* BOTONES */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        background-color: #ffffff;
        color: #475569;
        border: 1px solid #e2e8f0;
        padding: 8px 16px;
        font-size: 0.85rem;
    }
    .stButton>button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
        background-color: #f0f9ff;
    }

    /* KPI CARDS */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .kpi-title { font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 700; margin-bottom: 2px; }
    .kpi-value { font-size: 1.5rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
    .kpi-sub { font-size: 0.75rem; color: #94a3b8; }

    /* --- ESTILO DEL MODAL --- */
    div[data-testid="stDialog"] div[data-testid="stVerticalBlock"] {
        gap: 1rem !important; 
    }

    .umi-header-compact {
        margin-bottom: 15px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .umi-title-label { 
        font-size: 0.75rem; 
        text-transform: uppercase; 
        color: #64748b; 
        font-weight: 700; 
        letter-spacing: 0.5px;
    }
    .umi-id-text {
        font-family: 'Courier New', monospace; 
        font-size: 1.1rem; 
        font-weight: 700;
        color: #3b82f6; 
        background: #eff6ff; 
        padding: 4px 10px; 
        border-radius: 6px;
    }
    
    .data-section-title {
        font-size: 0.9rem; 
        font-weight: 800; 
        color: #334155; 
        margin-top: 15px; 
        margin-bottom: 8px;
        display: flex; 
        align-items: center; 
        gap: 8px; 
        text-transform: uppercase;
    }
    
    .info-row {
        display: flex; 
        justify-content: space-between; 
        padding: 6px 0; 
        border-bottom: 1px dashed #e2e8f0; 
        font-size: 0.9rem; 
        line-height: 1.5;
    }
    .info-label { color: #64748b; font-weight: 500; }
    .info-val { color: #1e293b; font-weight: 600; text-align: right; }

    /* Stats Grid dentro del Modal */
    .stat-container {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin: 10px 0;
    }
    .stat-mini-box {
        background: #f8fafc; 
        border: 1px solid #cbd5e1; 
        border-radius: 8px; 
        padding: 8px 4px; 
        text-align: center; 
        width: 32%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .stat-label { font-size: 0.7rem; color: #64748b; margin-top: 4px; font-weight: 500; }
    .stat-num { font-size: 1.1rem; font-weight: 800; color: #0f172a; line-height: 1; }

    .compact-divider { margin: 15px 0; border: 0; border-top: 1px solid #e2e8f0; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. FUNCIONES DE GEOCODIFICACIÓN & RECURSOS ---
MODELO_PATH = os.path.join('models', 'sipid_model.joblib')
GRID_ENCODER_PATH = os.path.join('models', 'grid_encoder.joblib')
MODALIDAD_ENCODER_PATH = os.path.join('models', 'modalidad_encoder.joblib')
DATOS_PROCESADOS_PATH = os.path.join('data', 'processed', 'datos_limpios_para_modelo.csv')
COORDENADAS_DEFAULT = [19.4326, -99.1332] 

@st.cache_resource
def cargar_recursos():
    try:
        modelo = joblib.load(MODELO_PATH)
        grid_encoder = joblib.load(GRID_ENCODER_PATH)
        mod_encoder = joblib.load(MODALIDAD_ENCODER_PATH) 
        return modelo, grid_encoder, mod_encoder
    except Exception as e:
        return None, None, None

@st.cache_data
def cargar_datos_historicos():
    try:
        df = pd.read_csv(DATOS_PROCESADOS_PATH)
        cols_fecha = [col for col in df.columns if 'fecha' in col.lower() or 'date' in col.lower()]
        if 'anio' not in df.columns:
            if cols_fecha:
                df['anio'] = pd.to_datetime(df[cols_fecha[0]], errors='coerce').dt.year
                df['anio'] = df['anio'].fillna(2024).astype(int)
            else: df['anio'] = 2024 
        return df
    except: return None

# --- GEOCODIFICACIÓN AVANZADA ---
@st.cache_data(ttl=3600, show_spinner=False)
def obtener_info_geografica(lat, lon):
    try:
        geolocator = Nominatim(user_agent="sipid_dashboard_v2")
        location = geolocator.reverse((lat, lon), exactly_one=True, language='es')
        if location:
            address = location.raw.get('address', {})
            return {
                "colonia": address.get('suburb') or address.get('neighbourhood') or address.get('residential') or "No ident.",
                "municipio": address.get('city') or address.get('town') or address.get('county') or "No ident.",
                "estado": address.get('state') or "S/D",
                "pais": address.get('country') or "México",
                "cp": address.get('postcode') or "S/D",
                "calle": address.get('road') or address.get('pedestrian') or "Vialidad sin nombre",
                "lat_fmt": f"{lat:.5f}",
                "lon_fmt": f"{lon:.5f}"
            }
    except (GeocoderTimedOut, Exception):
        return None
    return None

modelo, grid_encoder, mod_encoder = cargar_recursos()
df_historico = cargar_datos_historicos()

# --- 4. DIALOG / MODAL DE DETALLE UMI (FORENSE) ---
@st.dialog("📍 DETALLE FORENSE DE ZONA", width="large")
def mostrar_modal_umi(datos_umi, info_geo):
    c1, c2 = st.columns([1.5, 1]) 
    
    umi_id = datos_umi.get('Grid', datos_umi.get('Zona', 'N/A'))
    
    # --- CÁLCULO FORENSE EN TIEMPO REAL ---
    stats_forenses = {"total": 0, "hora_pico": "N/D", "dia_pico": "N/D"}
    
    if df_historico is not None:
        try:
            if 'grid_id' in df_historico.columns:
                df_umi = df_historico[df_historico['grid_id'] == umi_id]
            else:
                grid_num = grid_encoder.transform([umi_id])[0]
                df_umi = df_historico[df_historico['grid_num'] == grid_num]
            
            if not df_umi.empty:
                stats_forenses["total"] = len(df_umi)
                if 'hora_dia' in df_umi.columns:
                    mode_h = df_umi['hora_dia'].mode()
                    if not mode_h.empty: stats_forenses["hora_pico"] = f"{int(mode_h[0]):02d}:00"
                if 'dia_semana' in df_umi.columns:
                    dias_map = {0:'Lun', 1:'Mar', 2:'Mié', 3:'Jue', 4:'Vie', 5:'Sáb', 6:'Dom'}
                    mode_d = df_umi['dia_semana'].mode()
                    if not mode_d.empty: stats_forenses["dia_pico"] = dias_map.get(mode_d[0], "N/D")
        except: pass

    with c1:
        m_static = folium.Map(
            location=[datos_umi['Lat'], datos_umi['Lon']],
            zoom_start=16, 
            tiles="CartoDB positron",
            zoom_control=False, scrollWheelZoom=False, dragging=True 
        )
        folium.Polygon(
            locations=datos_umi["Boundary"],
            color="#ef4444", weight=3, fill=True, fill_opacity=0.25
        ).add_to(m_static)
        st_folium(m_static, width=700, height=550, key="static_map_modal")

    with c2:
        # 1. Cabecera
        st.markdown(f"""
            <div class='umi-header-compact'>
                <span class='umi-title-label'>Identificador UMI</span>
                <span class='umi-id-text'>{umi_id}</span>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. IA / Riesgo Actual
        st.markdown("<div class='data-section-title'>🧠 Estado Actual (IA)</div>", unsafe_allow_html=True)
        if datos_umi.get("Probabilidad", 0) > 0:
            st.markdown(f"""
                <div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:5px; padding: 5px 0;'>
                    <span style='font-size:1.8rem; font-weight:900; color:#ef4444; line-height:1;'>{datos_umi['Probabilidad']:.1%}</span>
                    <span style='font-size:0.8rem; color:#b91c1c; font-weight:700; background:#fee2e2; padding:4px 8px; border-radius:6px; letter-spacing:0.5px;'>RIESGO ACTIVO</span>
                </div>
            """, unsafe_allow_html=True)
            st.progress(datos_umi['Probabilidad'])
        else:
             st.markdown(f"""
                <div style='display:flex; align-items:center; gap:8px; padding: 5px 0;'>
                    <span style='font-size:1.8rem; font-weight:900; color:#3b82f6;'>{datos_umi['Delitos']}</span>
                    <span style='font-size:0.9rem; color:#64748b; font-weight:600;'>Incidentes en Filtro</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='compact-divider'></div>", unsafe_allow_html=True)

        # 3. Histórico Forense
        st.markdown("<div class='data-section-title'>📚 Histórico Total</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='stat-container'>
                <div class='stat-mini-box'>
                    <span class='stat-num'>{stats_forenses['total']}</span>
                    <span class='stat-label'>Total Hist.</span>
                </div>
                <div class='stat-mini-box'>
                    <span class='stat-num'>{stats_forenses['hora_pico']}</span>
                    <span class='stat-label'>Hora Crítica</span>
                </div>
                <div class='stat-mini-box'>
                    <span class='stat-num'>{stats_forenses['dia_pico']}</span>
                    <span class='stat-label'>Día Crítico</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='compact-divider'></div>", unsafe_allow_html=True)

        # 4. Datos Geográficos
        st.markdown("<div class='data-section-title'>🌍 Datos OSM</div>", unsafe_allow_html=True)
        
        def info_row(label, value):
            return f"<div class='info-row'><span class='info-label'>{label}</span><span class='info-val'>{value}</span></div>"

        html_geo = ""
        if info_geo:
            html_geo += info_row("Colonia", info_geo['colonia'])
            html_geo += info_row("Municipio", info_geo['municipio'])
            html_geo += info_row("Estado", info_geo['estado'])
            html_geo += info_row("C.P.", info_geo['cp'])
            html_geo += info_row("Vialidad", info_geo['calle'])
            html_geo += info_row("Coords", f"<span style='font-size:0.8rem; font-family:monospace; background:#f1f5f9; padding:2px 5px; border-radius:4px;'>{info_geo['lat_fmt']}, {info_geo['lon_fmt']}</span>")
        else:
            html_geo = "<p style='color:#f59e0b; font-size:0.9rem; margin:10px 0;'>⚠️ Sin conexión OSM.</p>"
        
        st.markdown(html_geo, unsafe_allow_html=True)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 15px 0;'>
            <h2 style='color: #0f172a; margin-bottom: 0; font-size: 1.5rem;'>🛡️ SIPID</h2>
            <span style='color: #3b82f6; font-size: 0.75rem; font-weight: 700; letter-spacing: 1.5px;'>CORE V2</span>
        </div>
    """, unsafe_allow_html=True)

    if 'tipo_visualizacion' not in st.session_state: st.session_state.tipo_visualizacion = None
    if 'data_viz_cache' not in st.session_state: st.session_state.data_viz_cache = [] 
    if 'centro_mapa_inicial' not in st.session_state: st.session_state.centro_mapa_inicial = COORDENADAS_DEFAULT
    if 'metricas_resumen' not in st.session_state: st.session_state.metricas_resumen = {}
    if 'df_graficas' not in st.session_state: st.session_state.df_graficas = pd.DataFrame()
    if 'show_modal_trigger' not in st.session_state: st.session_state.show_modal_trigger = False
    if 'selected_umi_data' not in st.session_state: st.session_state.selected_umi_data = None
    if 'last_map_click_signature' not in st.session_state: st.session_state.last_map_click_signature = None

    if modelo and mod_encoder:
        with st.expander("🎯 PREDICCIÓN AI", expanded=True):
            delito_sel = st.selectbox("Seleccionar Delito", list(mod_encoder.classes_))
            modalidad_num = mod_encoder.transform([delito_sel])[0]
            meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
            mes_sel = st.selectbox("Mes", meses, index=9)
            mes_num = meses.index(mes_sel) + 1
            dias_map = {0:'Lunes', 1:'Martes', 2:'Miércoles', 3:'Jueves', 4:'Viernes', 5:'Sábado', 6:'Domingo'}
            dia_nom = st.selectbox("Día de la semana", list(dias_map.values()), index=4)
            dia_num = list(dias_map.keys())[list(dias_map.values()).index(dia_nom)]
            opciones_horas = [f"{h:02d}:00" for h in range(24)]
            hora_txt = st.selectbox("Ventana Horaria", opciones_horas, index=19)
            hora_num = int(hora_txt.split(":")[0])
            btn_prediccion = st.button("GENERAR PREDICCIÓN")
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📂 HISTÓRICO POR DELITO", expanded=False):
            st.info(f"Análisis para: {delito_sel}")
            btn_tematico = st.button("GENERAR MAPA DE CALOR")
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🌍 PANORAMA GLOBAL", expanded=False):
            anio_sel = st.slider("Año de Análisis Global", 2020, 2025, 2024)
            btn_evolucion = st.button("GENERAR TENDENCIAS ANUALES")
    else:
        st.error("⚠️ Sistema Offline: Falta Engine.")

# --- 6. LÓGICA DE PROCESAMIENTO ---
def set_visualizacion(tipo, titulo, color, dato_prin, dato_sec, df_data):
    st.session_state.tipo_visualizacion = tipo
    st.session_state.df_graficas = df_data
    st.session_state.metricas_resumen = { "titulo": titulo, "color": color, "kpi1": dato_prin, "kpi2": dato_sec }

if btn_prediccion and modelo:
    if df_historico is not None:
        df_f = df_historico[df_historico['modalidad_num'] == modalidad_num].copy()
        set_visualizacion("prediccion", "RIESGO AI", "#0f172a", delito_sel, f"{dia_nom} {hora_txt}", df_f)

    with st.spinner("Consultando Red Neuronal..."):
        features = ['hora_dia', 'dia_semana', 'mes', 'modalidad_num']
        X_input = pd.DataFrame([[hora_num, dia_num, mes_num, modalidad_num]], columns=features)
        probs = modelo.predict_proba(X_input)[0]
        top_indices = probs.argsort()[::-1]
        
        data_viz = []
        try: st.session_state.centro_mapa_inicial = h3.cell_to_latlng(grid_encoder.inverse_transform([top_indices[0]])[0])
        except: pass

        for i, idx in enumerate(top_indices):
            try:
                grid_id = grid_encoder.inverse_transform([idx])[0]
                boundary = h3.cell_to_boundary(grid_id)
                lat, lon = h3.cell_to_latlng(grid_id)
                prob_val = probs[idx]
                
                if i < 10: color, nivel, risk, fill_op = '#ef4444', "CRÍTICO", True, 0.8
                elif i < 50: color, nivel, risk, fill_op = '#f97316', "ALTO", True, 0.6
                elif prob_val > 0.0001: color, nivel, risk, fill_op = '#cbd5e1', "BAJO", False, 0.4
                else: color, nivel, risk, fill_op = '#94a3b8', "NULO", False, 0.5
                
                data_viz.append({
                    "Ranking": i+1, "Nivel": nivel, "Probabilidad": prob_val,
                    "Ubicación": f"https://www.google.com/maps?q={lat},{lon}",
                    "Zona": grid_id, "Grid": grid_id,
                    "Lat": lat, "Lon": lon, "Boundary": boundary,
                    "Color": color, "FillOp": fill_op, "IsRisk": risk, "Delitos": 0
                })
            except: pass
        st.session_state.data_viz_cache = data_viz

if btn_tematico and df_historico is not None:
    df_f = df_historico[df_historico['modalidad_num'] == modalidad_num].copy()
    set_visualizacion("tematico", "HOTSPOTS", "#3b82f6", f"{len(df_f)} Casos", delito_sel, df_f)
    with st.spinner("Calculando densidad..."):
        if 'grid_id' not in df_f.columns:
             try: df_f['grid_id'] = grid_encoder.inverse_transform(df_f['grid_num'])
             except: pass
        conteo = df_f['grid_id'].value_counts()
        if not conteo.empty: st.session_state.centro_mapa_inicial = h3.cell_to_latlng(conteo.idxmax())
        data_viz = []
        colores = {"1-2": "#cbd5e1", "3-5": "#facc15", "6-10": "#f97316", "11-20": "#ef4444", ">20": "#991b1b"}
        
        for grid_id in list(grid_encoder.classes_):
            total = conteo.get(grid_id, 0)
            boundary = h3.cell_to_boundary(grid_id)
            lat, lon = h3.cell_to_latlng(grid_id)
            
            if total == 0: r, c, fill_op = "0", "#94a3b8", 0.5
            elif total <= 2: r, c, fill_op = "1-2", colores["1-2"], 0.7
            elif total <= 5: r, c, fill_op = "3-5", colores["3-5"], 0.7
            elif total <= 10: r, c, fill_op = "6-10", colores["6-10"], 0.7
            elif total <= 20: r, c, fill_op = "11-20", colores["11-20"], 0.7
            else: r, c, fill_op = ">20", colores[">20"], 0.7
            
            data_viz.append({
                "Grid": grid_id, "Zona": grid_id,
                "Delitos": total, "Rango": r,
                "Ubicación": f"https://www.google.com/maps?q={lat},{lon}",
                "Lat": lat, "Lon": lon, "Boundary": boundary, "Color": c, "FillOp": fill_op,
                "IsRisk": False, "Probabilidad": 0
            })
        st.session_state.data_viz_cache = data_viz

# --- AQUÍ ESTÁ LA NUEVA LÓGICA DE COLORES PROGRESIVA (AMARILLO -> ROJO) ---
if btn_evolucion and df_historico is not None:
    df_a = df_historico[df_historico['anio'] == anio_sel].copy()
    set_visualizacion("evolucion", "PANORAMA GLOBAL", "#8b5cf6", f"{len(df_a):,} Incidentes", f"Total Anual {anio_sel}", df_a)
    with st.spinner(f"Consolidando mapa de calor de riesgo {anio_sel}..."):
        if 'grid_id' not in df_a.columns:
            try: df_a['grid_id'] = grid_encoder.inverse_transform(df_a['grid_num'])
            except: pass
        conteo = df_a['grid_id'].value_counts()
        if not conteo.empty:
            st.session_state.centro_mapa_inicial = h3.cell_to_latlng(conteo.idxmax())
            data_viz = []
            
            # --- PALETA DE CALOR PROGRESIVA HACIA EL ROJO ---
            colores_evo = {
                "0": "#94a3b8",      # Gris (Sin actividad)
                "1-5": "#fcd34d",    # Amarillo (Ámbar bajo)
                "6-10": "#fb923c",   # Naranja Claro (Atención)
                "11-25": "#ea580c",  # Naranja Oscuro (Alerta)
                "26-50": "#dc2626",  # Rojo (Peligro)
                "51-100": "#b91c1c", # Rojo Intenso (Muy Alto)
                ">100": "#7f1d1d"    # Rojo Vino/Negro (Crítico)
            }
            
            for grid_id in list(grid_encoder.classes_):
                total = conteo.get(grid_id, 0)
                boundary = h3.cell_to_boundary(grid_id)
                lat, lon = h3.cell_to_latlng(grid_id)
                
                # Asignación de rangos
                if total == 0:
                    r, c, fill_op = "0", colores_evo["0"], 0.4
                elif total <= 5:
                    r, c, fill_op = "1-5", colores_evo["1-5"], 0.6
                elif total <= 10:
                    r, c, fill_op = "6-10", colores_evo["6-10"], 0.6
                elif total <= 25:
                    r, c, fill_op = "11-25", colores_evo["11-25"], 0.7
                elif total <= 50:
                    r, c, fill_op = "26-50", colores_evo["26-50"], 0.7
                elif total <= 100:
                    r, c, fill_op = "51-100", colores_evo["51-100"], 0.8
                else:
                    r, c, fill_op = ">100", colores_evo[">100"], 0.8
                
                data_viz.append({
                    "Grid": grid_id, "Zona": grid_id,
                    "Delitos": total, "Rango": r,
                    "Ubicación": f"https://www.google.com/maps?q={lat},{lon}",
                    "Lat": lat, "Lon": lon, "Boundary": boundary, "Color": c, "FillOp": fill_op,
                    "IsRisk": False, "Probabilidad": 0
                })
            st.session_state.data_viz_cache = data_viz

# --- 7. VISUALIZACIÓN PRINCIPAL ---
st.markdown("""
<div style='display:flex; justify-content:space-between; align-items:flex-end; padding: 20px 0 30px 0;'>
    <div>
        <h1 style='margin:0; font-size: 2.8rem; font-weight:800; color: #0f172a; letter-spacing: -1.5px;'>
            Dashboard <span style='color: #3b82f6;'>Inteligente</span>
        </h1>
        <p style='color: #64748b; margin:0; font-size: 1.1rem;'>Análisis predictivo con <span style='background:#e2e8f0; padding:2px 6px; border-radius:4px; font-size:0.9rem;'>Forensic UI V2</span></p>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.tipo_visualizacion:
    m = st.session_state.metricas_resumen
    col_k1, col_k2, col_k3 = st.columns(3)
    def kpi_html(title, value, subtitle):
        return f"""<div class="kpi-card"><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{subtitle}</div></div>"""
    with col_k1: st.markdown(kpi_html("PERSPECTIVA", m['titulo'], "Categoría del análisis actual"), unsafe_allow_html=True)
    with col_k2: st.markdown(kpi_html("DATO CLAVE", m['kpi1'], "Indicador de volumen o riesgo"), unsafe_allow_html=True)
    with col_k3: st.markdown(kpi_html("SEGMENTACIÓN", m['kpi2'], "Contexto de los datos filtrados"), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🗺️ Cartografía Interactiva", "📈 Estadísticas Avanzadas", "📋 Base de Datos"])

with tab1:
    if st.session_state.tipo_visualizacion and st.session_state.data_viz_cache:
        col_mapa, col_lateral = st.columns([2.5, 1])
        df_viz = pd.DataFrame(st.session_state.data_viz_cache)
        with col_lateral:
            st.markdown("##### 📍 Análisis de Zona")
            
            if st.session_state.tipo_visualizacion == "tematico":
                df_pie = df_viz[df_viz["Delitos"] > 0]["Rango"].value_counts().reset_index()
                df_pie.columns = ["Rango", "Cantidad"]
                colores_pie = {"1-2": "#cbd5e1", "3-5": "#facc15", "6-10": "#f97316", "11-20": "#ef4444", ">20": "#991b1b"}
                pie = alt.Chart(df_pie).mark_arc(innerRadius=45, cornerRadius=10).encode(
                    theta="Cantidad", color=alt.Color("Rango", scale=alt.Scale(domain=list(colores_pie.keys()), range=list(colores_pie.values())), legend=None), tooltip=["Rango", "Cantidad"]
                ).properties(height=180)
                st.altair_chart(pie, use_container_width=True)

            # --- CORRECCIÓN GRÁFICA PARA NUEVOS RANGOS (EVOLUCIÓN) ---
            elif st.session_state.tipo_visualizacion == "evolucion":
                df_pie = df_viz[df_viz["Delitos"] > 0]["Rango"].value_counts().reset_index()
                df_pie.columns = ["Rango", "Cantidad"]
                # Debe coincidir con los colores definidos arriba
                colores_evo_pie = {
                    "1-5": "#fcd34d", "6-10": "#fb923c", "11-25": "#ea580c",
                    "26-50": "#dc2626", "51-100": "#b91c1c", ">100": "#7f1d1d"
                }
                pie = alt.Chart(df_pie).mark_arc(innerRadius=45, cornerRadius=10).encode(
                    theta="Cantidad", color=alt.Color("Rango", scale=alt.Scale(domain=list(colores_evo_pie.keys()), range=list(colores_evo_pie.values())), legend=None), tooltip=["Rango", "Cantidad"]
                ).properties(height=180)
                st.altair_chart(pie, use_container_width=True)

            if st.session_state.tipo_visualizacion == "prediccion":
                df_top = df_viz[df_viz["IsRisk"] == True].head(10).copy()
                cols_view = ["Ranking", "Probabilidad", "Ubicación"]
            else:
                df_top = df_viz.sort_values("Delitos", ascending=False).head(10).copy()
                cols_view = ["Delitos", "Rango", "Ubicación"]
            st.dataframe(df_top[cols_view], column_config={"Probabilidad": st.column_config.ProgressColumn("Nivel", format="%.2f", min_value=0, max_value=1.0), "Ubicación": st.column_config.LinkColumn("Mapa", display_text="📍"), "Delitos": st.column_config.NumberColumn("Casos")}, use_container_width=True, hide_index=True)
            st.caption("ℹ️ Haz clic en un hexágono del mapa para ver el análisis forense de la zona.")

        with col_mapa:
            loc = st.session_state.centro_mapa_inicial
            m = folium.Map(location=loc, zoom_start=14, tiles="CartoDB positron", zoom_control=False)
            for item in st.session_state.data_viz_cache:
                zone_key = item.get('Zona', item.get('Grid', 'N/A'))
                txt = f"Zona: {zone_key} " 
                folium.Polygon(locations=item["Boundary"], color=item["Color"], fill=True, fill_color=item["Color"], fill_opacity=item.get("FillOp",0.6), weight=1, popup=txt, tooltip=zone_key).add_to(m)
            map_data = st_folium(m, width="100%", height=550, key="mapa_v2")
            
            clicked_zone_id = map_data.get("last_object_clicked_tooltip")
            clicked_point = map_data.get("last_object_clicked")
            click_signature = None
            if clicked_zone_id and isinstance(clicked_point, dict):
                lat = clicked_point.get("lat")
                lng = clicked_point.get("lng")
                if lat is not None and lng is not None:
                    click_signature = f"{clicked_zone_id}|{lat:.6f}|{lng:.6f}"
            elif clicked_zone_id:
                click_signature = str(clicked_zone_id)

            if click_signature and click_signature != st.session_state.last_map_click_signature:
                st.session_state.last_map_click_signature = click_signature
                zone_data = next((item for item in st.session_state.data_viz_cache if item.get("Zona") == clicked_zone_id or item.get("Grid") == clicked_zone_id), None)
                if zone_data:
                    st.session_state.selected_umi_data = zone_data
                    st.session_state.show_modal_trigger = True
                    
            if st.session_state.show_modal_trigger and st.session_state.selected_umi_data:
                umi_data = st.session_state.selected_umi_data
                with st.spinner("Conectando satélite y OSM..."):
                     info_geo = obtener_info_geografica(umi_data['Lat'], umi_data['Lon'])
                mostrar_modal_umi(umi_data, info_geo)
                st.session_state.show_modal_trigger = False
    else:
        st.markdown("""<div style='height: 400px; display: flex; align-items: center; justify-content: center; background: #f8fafc; border-radius: 20px; border: 2px dashed #e2e8f0;'><div style='text-align: center; color: #94a3b8;'><p style='font-size: 3rem;'>🎯</p><p>Configura los parámetros en el panel lateral para iniciar el motor predictivo o global.</p></div></div>""", unsafe_allow_html=True)

with tab2:
    st.markdown("#### Patrones de Comportamiento")
    df_g = st.session_state.df_graficas
    if not df_g.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.caption("Incidencia por Día")
            d_counts = df_g['dia_semana'].value_counts().reset_index()
            d_counts.columns = ['d', 'c']
            d_counts['n'] = d_counts['d'].map({0:'Lun', 1:'Mar', 2:'Mié', 3:'Jue', 4:'Vie', 5:'Sáb', 6:'Dom'})
            chart = alt.Chart(d_counts).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#0f172a").encode(x=alt.X('n', sort=['Lun','Mar','Mié','Jue','Vie','Sáb','Dom'], title=None), y=alt.Y('c', title=None), tooltip=['n','c']).properties(height=280)
            st.altair_chart(chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.caption("Densidad Horaria")
            h_counts = df_g['hora_dia'].value_counts().reset_index()
            h_counts.columns = ['h', 'c']
            area = alt.Chart(h_counts).mark_area(line={'color':'#3b82f6'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop('#3b82f6',0), alt.GradientStop('white',1)], x1=1, y1=1, x2=1, y2=0)).encode(x=alt.X('h', title='Hora'), y=alt.Y('c', title=None)).properties(height=280)
            st.altair_chart(area, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    if st.session_state.data_viz_cache:
        df_viz_plot = pd.DataFrame(st.session_state.data_viz_cache)
        if st.session_state.tipo_visualizacion == "prediccion":
            st.markdown("#### 🎯 Top 20 Zonas de Mayor Riesgo (UMIs)")
            top_20 = df_viz_plot.sort_values("Probabilidad", ascending=False).head(20)
            if 'Zona' in top_20.columns: x_ax = 'Zona:N'
            else: x_ax = 'Grid:N'
            chart_umi = alt.Chart(top_20).mark_bar(cornerRadiusBottomRight=10, cornerRadiusTopRight=10).encode(x=alt.X('Probabilidad:Q', title="Nivel de Probabilidad", axis=alt.Axis(format='%')), y=alt.Y(x_ax, sort='-x', title="Identificador de Zona (UMI)"), color=alt.Color('Probabilidad:Q', scale=alt.Scale(scheme='reds'), legend=None), tooltip=['Ranking', x_ax, alt.Tooltip('Probabilidad:Q', format='.2%'), 'Nivel']).properties(height=500)
        else:
            st.markdown("#### 📊 Top 20 Zonas por Volumen de Incidentes")
            top_20 = df_viz_plot.sort_values("Delitos", ascending=False).head(20)
            if 'Grid' in top_20.columns: x_ax = 'Grid:N'
            else: x_ax = 'Zona:N'
            chart_umi = alt.Chart(top_20).mark_bar(cornerRadiusBottomRight=10, cornerRadiusTopRight=10).encode(x=alt.X('Delitos:Q', title="Número de Casos Registrados"), y=alt.Y(x_ax, sort='-x', title="Identificador de Zona (UMI)"), color=alt.Color('Delitos:Q', scale=alt.Scale(scheme='reds'), legend=None), tooltip=[x_ax, 'Delitos', 'Rango']).properties(height=500)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.altair_chart(chart_umi, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.data_viz_cache:
        df_viz = pd.DataFrame(st.session_state.data_viz_cache)
        st.markdown("#### Inspección de Datos Crudos")
        st.dataframe(df_viz.drop(columns=['Boundary', 'FillOp'], errors='ignore'), column_config={"Ubicación": st.column_config.LinkColumn("Enlace Geográfico")}, use_container_width=True, height=500)
