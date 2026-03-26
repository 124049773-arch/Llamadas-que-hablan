import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sqlite3
from datetime import datetime
import plotly.graph_objects as go
import numpy as np

WIDTH = 650
HEIGHT = 450

st.set_page_config(page_title="Dashb Línea de Mujeres", layout="wide")

st.markdown("""
<style>
.stApp {background-color: #F9F5FF;}
[data-testid="stSidebar"] {background-color: #E6CCFF;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {color: #581C87;}
.stMultiSelect div[data-baseweb="select"] {background-color: #F3E8FF; border-radius: 10px;}
span[data-baseweb="tag"] {background-color: #9333EA !important; color: white !important; border-radius: 8px;}
span[data-baseweb="tag"] svg {fill: white !important;}
span[data-baseweb="tag"]:hover {background-color: #7E22CE !important;}
h1 {color: #6B21A8;}
</style>
""", unsafe_allow_html=True)

st.title("Llamadas que hablan Línea de Mujeres CDMX")
st.title("La cicatriz es la prueba de que sobreviviste, pero tu brillo es la prueba de que venciste")
st.markdown("Visualización dinámica de los reportes de atención.")

# ==================== CONFIGURACIÓN DE BASE DE DATOS ====================
def init_database():
    """Inicializa la base de datos SQLite"""
    conn = sqlite3.connect('cuestionario_mujeres.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS respuestas_cuestionario (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TIMESTAMP,
        edad_grupo TEXT,
        situacion TEXT,
        frecuencia TEXT,
        relacion TEXT,
        hablado_alguien TEXT
    )''')
    
    conn.commit()
    conn.close()

def guardar_respuesta(datos):
    """Guarda las respuestas del cuestionario en la base de datos"""
    try:
        conn = sqlite3.connect('cuestionario_mujeres.db')
        c = conn.cursor()
        
        c.execute('''INSERT INTO respuestas_cuestionario 
                    (fecha, edad_grupo, situacion, frecuencia, relacion, hablado_alguien)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                  (datetime.now(), 
                   datos['edad_grupo'],
                   datos['situacion'],
                   datos['frecuencia'],
                   datos['relacion'],
                   datos['hablado_alguien']))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error al guardar: {e}")
        return False

def cargar_respuestas_cuestionario():
    """Carga las respuestas del cuestionario para análisis"""
    try:
        conn = sqlite3.connect('cuestionario_mujeres.db')
        df = pd.read_sql_query("SELECT * FROM respuestas_cuestionario", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# Inicializar base de datos
init_database()
# ==================== FIN CONFIGURACIÓN BASE DE DATOS ====================

@st.cache_data
def load_data():
    df = pd.read_csv("linea-mujeres-cdmx.csv", encoding="latin1")
    df.columns = df.columns.str.lower().str.strip()
    return df

df = load_data()

# ==================== FILTROS ====================
st.sidebar.header("Filtros")

# Filtro por Estado
estados_disponibles = df['estado_usuaria'].unique()
estado = st.sidebar.multiselect(
    "Selecciona Estado:",
    options=estados_disponibles,
    default=estados_disponibles[:3] if len(estados_disponibles) > 3 else estados_disponibles
)

# Filtrar por estado primero
if estado:
    df_filtrado_estado = df[df['estado_usuaria'].isin(estado)]
else:
    df_filtrado_estado = df

# Filtro por Municipio
municipios_disponibles = df_filtrado_estado['municipio_usuaria'].unique()
municipio = st.sidebar.multiselect(
    "Selecciona Municipio:",
    options=municipios_disponibles,
    default=municipios_disponibles[:5] if len(municipios_disponibles) > 5 else municipios_disponibles
)

# Aplicar filtro de municipio
if municipio:
    df_selection = df_filtrado_estado[df_filtrado_estado['municipio_usuaria'].isin(municipio)]
else:
    df_selection = df_filtrado_estado

col1, col2, col3 = st.columns(3)
col1.metric("Total Reportes", f"{len(df_selection):,}")
edad_promedio = pd.to_numeric(df_selection['edad'], errors='coerce').mean()
col2.metric("Edad Promedio", f"{edad_promedio:.0f}" if not pd.isna(edad_promedio) else "N/A")
col3.metric("Municipios", f"{len(municipio) if municipio else len(df_selection['municipio_usuaria'].unique())}")

# ==================== GRÁFICA 1: DISTRIBUCIÓN POR OCUPACIÓN ====================
c1, c2 = st.columns([2,1])

with c1:
    st.subheader("Distribución por Ocupación")
    fig_ocupacion = px.pie(df_selection, names='ocupacion', hole=0.6,
        color_discrete_sequence=["#E6CCFF","#D8B4FE","#C084FC","#A855F7","#9333EA","#7E22CE"])
    fig_ocupacion.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_ocupacion, use_container_width=False)

with c2:
    st.subheader("Análisis gráfico")
    st.write("""
    - La gráfica muestra la distribución de las ocupaciones de las usuarias que realizaron llamadas.
    - Se observa qué grupos ocupacionales tienen mayor presencia en los reportes.
    - Las porciones más grandes indican los sectores laborales con más casos.
    - Esto permite focalizar campañas de prevención en sectores específicos.
    """)

# ==================== GRÁFICA 2: ATENCIONES POR MES ====================
c3, c4 = st.columns([2,1])

with c3:
    st.subheader("Atenciones por Mes")
    mes_counts = df_selection['mes_alta'].value_counts().reset_index()
    mes_counts.columns = ['mes', 'total']
    fig_mes = px.bar(mes_counts.sort_values(by='mes'), x='mes', y='total',
        labels={'mes': 'Mes del Año', 'total': 'Número de llamadas'},
        color_discrete_sequence=['#9333EA'])
    fig_mes.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_mes, use_container_width=False)

with c4:
    st.subheader("Análisis gráfico")
    st.write("""
    - La gráfica muestra la distribución de llamadas por cada mes del año.
    - Se identifican los meses con mayor y menor número de reportes.
    - Los picos más altos indican épocas de mayor demanda de atención.
    - Permite planificar recursos según la demanda estacional.
    """)

# ==================== GRÁFICA 3: DISTRIBUCIÓN DE EDADES ====================
c5, c6 = st.columns([2,1])

with c5:
    st.subheader("Distribución de Edades")
    bins = st.slider("Número de intervalos (bins)", 5, 50, 20, key="bins_edad")
    fig_edad = px.histogram(df_selection, x="edad", nbins=bins,
        title="Distribución de Edades de las Usuarias", color_discrete_sequence=['#FFA200'])
    fig_edad.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_edad, use_container_width=False)

with c6:
    st.subheader("Análisis gráfico")
    st.write("""
    - La gráfica muestra la concentración de edades de las usuarias que reportan.
    - Se observa que la mayoría de las personas se concentran entre los 30 y 50 años.
    - Hay menos casos en edades muy jóvenes y en edades muy avanzadas.
    - El núcleo más fuerte está en edades medias, los extremos son poco frecuentes.
    """)

# ==================== GRÁFICA 4: FRECUENCIA POR ESTADO CIVIL ====================
c7, c8 = st.columns([2,1])

with c7:
    if 'estado_civil' in df.columns:
        st.subheader("Frecuencia por estado civil")
        conteo_ec = df['estado_civil'].value_counts().reset_index()
        conteo_ec.columns = ['estado_civil', 'total']
        fig_ec = px.bar(conteo_ec, x='estado_civil', y='total', color_discrete_sequence=["#9333EA"])
        fig_ec.update_layout(width=WIDTH, height=HEIGHT, xaxis_tickangle=45)
        st.plotly_chart(fig_ec, use_container_width=False)

with c8:
    st.subheader("Análisis gráfico")
    st.write("""
    - La gráfica muestra la distribución por estado civil de las mujeres que reportan.
    - Se observa que la mayoría son mujeres solteras con alrededor de 250 mil registros.
    - Le siguen las mujeres casadas con aproximadamente 150 mil casos.
    - Las de unión libre tienen cerca de 50 mil casos registrados.
    """)

# ==================== GRÁFICA 5: EVOLUCIÓN MENSUAL DE LLAMADAS ====================
c9, c10 = st.columns([2,1])

with c9:
    st.subheader("Evolución mensual de llamadas")
    df_temp = df.copy()
    df_temp['fecha_alta'] = pd.to_datetime(df_temp['fecha_alta'], errors='coerce')
    df_temp = df_temp.dropna(subset=['fecha_alta'])
    df_temp['anio_mes'] = df_temp['fecha_alta'].dt.to_period('M').astype(str)
    llamadas_por_mes = df_temp.groupby('anio_mes').size().reset_index()
    llamadas_por_mes.columns = ['anio_mes', 'total']
    fig_ev = px.line(llamadas_por_mes, x='anio_mes', y='total', markers=True)
    fig_ev.update_layout(width=WIDTH, height=HEIGHT, xaxis_tickangle=90)
    st.plotly_chart(fig_ev, use_container_width=False)

with c10:
    st.subheader("Análisis gráfico")
    st.write("""
    - La gráfica muestra la evolución de las llamadas a lo largo del tiempo.
    - Se observa un aumento al inicio, luego se mantuvieron con altibajos.
    - Finalmente se ve una caída significativa en los últimos períodos.
    - Permite identificar tendencias y evaluar el impacto de intervenciones.
    """)

# ==================== GRÁFICA 6: CLUSTERS EDAD VS SERVICIO ====================
c11, c12 = st.columns([2,1])

with c11:
    st.subheader("Clusters de llamadas (Edad vs Servicio)")
    df_num = df.select_dtypes(include=['int64','float64']).fillna(df.median(numeric_only=True))
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_num)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(datos_escalados)
    if 'edad' in df.columns and 'servicio' in df.columns:
        fig_cl = px.scatter(df, x='edad', y='servicio', color='cluster', color_continuous_scale='viridis')
        fig_cl.update_layout(width=WIDTH, height=HEIGHT)
        st.plotly_chart(fig_cl, use_container_width=False)

with c12:
    st.subheader("Análisis gráfico")
    st.write("""
    - El eje horizontal indica la edad de las mujeres que llamaron.
    - El eje vertical indica el tipo de asesoría requerida.
    - Los puntos representan las llamadas y el color indica a qué grupo de edad pertenecen.
    - Permite identificar patrones de servicio según rangos de edad.
    """)

# ==================== ANÁLISIS DE TEMÁTICAS ====================
st.header("Análisis de Temáticas")

# Preparar datos de temáticas
columnas_tematicas = ['tematica_1', 'tematica_2', 'tematica_3', 'tematica_4', 'tematica_5', 'tematica_6', 'tematica_7']
tematicas_existentes = [col for col in columnas_tematicas if col in df.columns]

if tematicas_existentes:
    # Crear versión expandida para análisis
    df_temp = df_selection.copy()
    for col in tematicas_existentes:
        df_temp[col] = df_temp[col].fillna('No especificado')
    
    df_temp['tematicas_lista'] = df_temp[tematicas_existentes].apply(lambda x: x.tolist(), axis=1)
    df_exploded = df_temp.explode('tematicas_lista')
    df_exploded = df_exploded[df_exploded['tematicas_lista'] != 'No especificado']
    df_exploded = df_exploded.rename(columns={'tematicas_lista': 'tematica'})
    
    # Crear grupos de edad
    df_exploded['edad'] = pd.to_numeric(df_exploded['edad'], errors='coerce')
    df_exploded['grupo_edad'] = pd.cut(
        df_exploded['edad'],
        bins=[0, 18, 25, 35, 45, 100],
        labels=['<18 años', '18-25 años', '26-35 años', '36-45 años', '>45 años']
    )
    
    # GRÁFICA 7: TOP 15 TEMÁTICAS MÁS REPORTADAS
    c_tem1, c_tem2 = st.columns([2,1])
    
    with c_tem1:
        st.subheader("Top 15 temáticas más reportadas")
        
        top_tematicas = df_exploded['tematica'].value_counts().head(15)
        
        fig_top_tematicas = px.bar(
            x=top_tematicas.values, 
            y=top_tematicas.index,
            orientation='h',
            labels={'x': 'Número de casos', 'y': 'Temática'},
            color=top_tematicas.values,
            color_continuous_scale='Purples_r'
        )
        fig_top_tematicas.update_layout(width=WIDTH, height=HEIGHT, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top_tematicas, use_container_width=False)
    
    with c_tem2:
        st.subheader("Análisis gráfico")
        st.write("""
        - La gráfica muestra las 15 problemáticas más frecuentes reportadas por las usuarias.
        - Las barras más largas y de color morado más oscuro representan los problemas más comunes.
        - Se observa una clara concentración en las primeras 3-4 temáticas, lo que indica problemas prioritarios.
        - La temática con mayor número de casos debe ser la principal atención de los programas de apoyo.
        """)
    
    # GRÁFICA 8: DISTRIBUCIÓN DE TEMÁTICAS POR GRUPO DE EDAD
    c_tem3, c_tem4 = st.columns([2,1])
    
    with c_tem3:
        st.subheader("Distribución de temáticas por grupo de edad")
        
        # Seleccionar top 8 temáticas
        top8 = df_exploded['tematica'].value_counts().head(8).index
        df_top8 = df_exploded[df_exploded['tematica'].isin(top8)]
        
        # Crear tabla de contingencia
        edad_tematica = pd.crosstab(df_top8['tematica'], df_top8['grupo_edad'])
        
        # Colores de morado para grupos de edad
        colores_morados = ['#4A0E4E', '#6B2E6B', '#8B4B8B', '#AA6EAA', '#C999C9']
        
        fig_tematica_edad = px.bar(
            edad_tematica,
            labels={'value': 'Número de casos', 'tematica': 'Temática', 'variable': 'Grupo de Edad'},
            color_discrete_sequence=colores_morados,
            barmode='stack'
        )
        fig_tematica_edad.update_layout(width=WIDTH, height=HEIGHT, xaxis_tickangle=45)
        st.plotly_chart(fig_tematica_edad, use_container_width=False)
    
    with c_tem4:
        st.subheader("Análisis gráfico")
        st.write("""
        - La gráfica muestra cómo se distribuyen las problemáticas según la edad de las usuarias.
        - El color morado más oscuro (#4A0E4E) representa al grupo de menor edad (<18 años).
        - El color morado más claro (#C999C9) representa al grupo de mayor edad (>45 años).
        - Se observa qué problemáticas afectan más a jóvenes y cuáles a adultas mayores.
        - Los colores equilibrados indican problemáticas que afectan a todas las edades.
        """)

else:
    st.warning("No se encontraron columnas de temáticas en los datos")

# ==================== ANÁLISIS DE ESCOLARIDAD ====================
st.header("Análisis de Escolaridad")

if 'escolaridad' in df.columns:
    # Preparar datos de temáticas para análisis con escolaridad
    columnas_tematicas = ['tematica_1', 'tematica_2', 'tematica_3', 'tematica_4', 'tematica_5', 'tematica_6', 'tematica_7']
    tematicas_existentes_esc = [col for col in columnas_tematicas if col in df.columns]
    
    if tematicas_existentes_esc:
        df_temp_esc = df_selection.copy()
        for col in tematicas_existentes_esc:
            df_temp_esc[col] = df_temp_esc[col].fillna('No especificado')
        
        df_temp_esc['tematicas_lista'] = df_temp_esc[tematicas_existentes_esc].apply(lambda x: x.tolist(), axis=1)
        df_exploded_esc = df_temp_esc.explode('tematicas_lista')
        df_exploded_esc = df_exploded_esc[df_exploded_esc['tematicas_lista'] != 'No especificado']
        df_exploded_esc = df_exploded_esc.rename(columns={'tematicas_lista': 'tematica'})
    else:
        df_exploded_esc = df_selection.copy()
    
    # GRÁFICA 9: ESCOLARIDAD VS TIPO DE VIOLENCIA
    c_esc1, c_esc2 = st.columns([2,1])
    
    with c_esc1:
        st.subheader("Escolaridad vs Tipo de Violencia")
        
        if 'tematica' in df_exploded_esc.columns:
            # Crear tabla de contingencia
            escolaridad_violencia = pd.crosstab(df_exploded_esc['tematica'], df_exploded_esc['escolaridad'])
            
            # Seleccionar top 10 temáticas
            top10_esc = escolaridad_violencia.sum(axis=1).sort_values(ascending=False).head(10).index
            escolaridad_violencia_top = escolaridad_violencia.loc[top10_esc]
            
            # Heatmap
            fig_esc_viol = px.imshow(
                escolaridad_violencia_top,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Purples',
                title="Relación: Escolaridad vs Tipo de Violencia",
                labels={'x': 'Nivel Educativo', 'y': 'Tipo de Violencia', 'color': 'Número de casos'}
            )
            fig_esc_viol.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig_esc_viol, use_container_width=False)
        else:
            st.info("No hay datos de temáticas para este análisis")
    
    with c_esc2:
        st.subheader("Análisis gráfico")
        st.write("""
        - La gráfica muestra qué tipos de violencia son más comunes según el nivel educativo.
        - Los colores más oscuros indican mayor concentración de casos en esa combinación.
        - Permite identificar si la violencia psicológica es más reportada por mujeres con estudios superiores.
        - Ayuda a diseñar campañas de prevención adaptadas a cada nivel educativo.
        """)
    
    # GRÁFICA 10: ESTADO CIVIL POR NIVEL EDUCATIVO
    c_esc3, c_esc4 = st.columns([2,1])
    
    with c_esc3:
        st.subheader("Estado civil por nivel educativo")
        
        if 'estado_civil' in df_exploded_esc.columns:
            escolaridad_estado = pd.crosstab(df_exploded_esc['escolaridad'], df_exploded_esc['estado_civil'])
            
            fig_esc_estado = px.bar(
                escolaridad_estado,
                barmode='stack',
                title="Estado Civil por Nivel Educativo",
                labels={'value': 'Número de casos', 'escolaridad': 'Nivel Educativo', 'variable': 'Estado Civil'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_esc_estado.update_layout(width=WIDTH, height=HEIGHT, xaxis_tickangle=45)
            st.plotly_chart(fig_esc_estado, use_container_width=False)
        else:
            st.info("No hay datos de estado civil para este análisis")
    
    with c_esc4:
        st.subheader("Análisis gráfico")
        st.write("""
        - La gráfica muestra la distribución de estados civiles según el nivel educativo.
        - Permite identificar si mujeres con estudios superiores tienen diferentes patrones de estado civil.
        - Se observa si hay mayor proporción de solteras en niveles educativos altos.
        - Ayuda a entender el contexto socio-familiar según el nivel de estudios.
        """)

else:
    st.warning("No se encontró la columna 'escolaridad' en los datos")
    st.info("Para agregar este análisis, asegúrate de que tu archivo CSV contenga la columna 'escolaridad'")

# ==================== CUESTIONARIO ====================
st.markdown("---")
st.header("Encuesta")
st.title("Cuéntanos qué fue lo que sucedió ese día")
st.markdown("Responde lo más sincera posible")

with st.form(key="cuestionario_form"):
    edad_grupo = st.selectbox(
        "¿Qué edad tienes?",
        ["Menor de 10", "10-15", "15-25", "25-35", "35-45", "Mayor de 45"]
    )
    
    situacion = st.selectbox(
        "¿Has experimentado alguna situación?",
        ["Abuso sexual", "Violencia Familiar", "Abuso de confianza", "Violación en la escuela o trabajo", "Otros"]
    )
    
    frecuencia = st.selectbox(
        "¿Con qué frecuencia ocurre?",
        ["Ocurrió una vez", "De vez en cuando", "Frecuentemente", "Me está pasando ahora"]
    )
    
    relacion = st.selectbox(
        "Relación con la persona",
        ["Pareja", "Familiar", "Trabajo", "Otro"]
    )
    
    hablado_alguien = st.selectbox(
        "¿Has hablado con alguien?",
        ["Sí", "No"]
    )
    
    submitted = st.form_submit_button("Enviar")
    
    if submitted:
        datos_respuesta = {
            'edad_grupo': edad_grupo,
            'situacion': situacion,
            'frecuencia': frecuencia,
            'relacion': relacion,
            'hablado_alguien': hablado_alguien
        }
        
        if guardar_respuesta(datos_respuesta):
            st.success("¡Gracias por tu confianza! Tu respuesta ha sido guardada.")
            st.balloons()
        else:
            st.error("Hubo un error al guardar tu respuesta. Por favor, intenta de nuevo.")

if st.button("Necesitas ayuda"):
    st.warning("""Llama al: 800 10 84 053 o 079  Recuerda no estas sola. Puedes acudar a las siguientes sedes, no tengas miedo de hablar: 
Secretaría de las Mujeres
Prolongación Corregidora Sur 210, 76074 Querétaro
442 215 3404         
Secretaría de la Mujer del Municipio de Querétaro
Galaxia 543, 76085 Santiago de Querétaro, Querétaro
442 238 7700
Secretaría Municipal de la Mujer Corregidora Querétaro
Calle Monterrey, 76902 Corregidora, Querétaro""")

df_respuestas = cargar_respuestas_cuestionario()
if not df_respuestas.empty:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Estadísticas del Cuestionario")
    st.sidebar.metric("Respuestas recibidas", len(df_respuestas))
    st.sidebar.metric("Última respuesta", df_respuestas['fecha'].max().split()[0] if not df_respuestas.empty else "N/A")