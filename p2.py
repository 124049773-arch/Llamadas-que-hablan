import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sqlite3
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import zipfile

WIDTH = 650
HEIGHT = 450

st.set_page_config(page_title="Women's Line Dashboard", layout="wide")

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

st.title("Calls that Speak - Women's Line CDMX")
st.title("The scar is proof you survived, but your shine is proof you conquered")
st.markdown("Dynamic visualization of care reports.")

# ==================== DATABASE CONFIGURATION ====================
def init_database():
    conn = sqlite3.connect('questionnaire_women.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS questionnaire_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP, age_group TEXT, situation TEXT, 
        frequency TEXT, relationship TEXT, talked_to_someone TEXT
    )''')
    conn.commit()
    conn.close()

def save_response(data):
    try:
        conn = sqlite3.connect('questionnaire_women.db')
        c = conn.cursor()
        c.execute('''INSERT INTO questionnaire_responses 
                    (timestamp, age_group, situation, frequency, relationship, talked_to_someone)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                  (datetime.now(), data['age_group'], data['situation'], 
                   data['frequency'], data['relationship'], data['talked_to_someone']))
        conn.commit()
        conn.close()
        return True
    except: return False

def load_questionnaire_responses():
    try:
        conn = sqlite3.connect('questionnaire_women.db')
        df = pd.read_sql_query("SELECT * FROM questionnaire_responses", conn)
        conn.close()
        return df
    except: return pd.DataFrame()

init_database()

# ==================== LOAD DATA CORREGIDO ====================
@st.cache_data
def load_data():
    zip_path = "linea-mujeres-cdmx.zip" 
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith('.csv')]
            if not csv_files:
                st.error("El ZIP no tiene un CSV adentro.")
                st.stop()
            with z.open(csv_files[0]) as f:
                # sep=None detecta automáticamente si el archivo usa , o ;
                df = pd.read_csv(f, encoding="latin1", sep=',', on_bad_lines='skip')
        
        # Limpieza de nombres de columnas: quita espacios, comillas y pasa a minúsculas
        df.columns = [str(c).lower().strip().replace('"', '').replace("'", "") for c in df.columns]
        
        if 'edad' in df.columns:
            df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error crítico al leer los datos: {e}")
        st.stop()

df = load_data()

# ==================== FILTERS ====================
st.sidebar.header("Filters")

# Filtros usando los nombres limpios
available_states = df['estado_usuaria'].unique()
state = st.sidebar.multiselect("Select State:", options=available_states, 
                               default=available_states[:3] if len(available_states) > 3 else available_states)

filtered_df_state = df[df['estado_usuaria'].isin(state)] if state else df

available_municipalities = filtered_df_state['municipio_usuaria'].unique()
municipality = st.sidebar.multiselect("Select Municipality:", options=available_municipalities,
                                      default=available_municipalities[:5] if len(available_municipalities) > 5 else available_municipalities)

df_selection = filtered_df_state[filtered_df_state['municipio_usuaria'].isin(municipality)] if municipality else filtered_df_state

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Reports", f"{len(df_selection):,}")
avg_age = pd.to_numeric(df_selection['edad'], errors='coerce').mean()
col2.metric("Average Age", f"{avg_age:.0f}" if not pd.isna(avg_age) else "N/A")
col3.metric("Municipalities", f"{len(municipality) if municipality else len(df_selection['municipio_usuaria'].unique())}")

# ==================== GRAPHS AND ANALYSES ====================

# GRAPH 1
c1, c2 = st.columns([2,1])
with c1:
    st.subheader("Distribution by Occupation")
    fig_occ = px.pie(df_selection, names='ocupacion', hole=0.6, color_discrete_sequence=["#E6CCFF","#D8B4FE","#C084FC","#A855F7","#9333EA","#7E22CE"])
    fig_occ.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_occ)
with c2:
    st.subheader("Graph Analysis")
    st.write("""
    - The graph shows the distribution of occupations of users who made calls.
    - It shows which occupational groups have the highest presence in reports.
    - Larger portions indicate labor sectors with more cases.
    - This helps focus prevention campaigns on specific sectors.
    """)

# GRAPH 2
c3, c4 = st.columns([2,1])
with c3:
    st.subheader("Attentions by Month")
    month_counts = df_selection['mes_alta'].value_counts().reset_index()
    month_counts.columns = ['month', 'total']
    fig_month = px.bar(month_counts.sort_values(by='month'), x='month', y='total', color_discrete_sequence=['#9333EA'])
    fig_month.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_month)
with c4:
    st.subheader("Graph Analysis")
    st.write("""
    - The graph shows call distribution for each month of the year.
    - Identifies months with the highest and lowest number of reports.
    - Highest peaks indicate times of greatest demand for attention.
    - Helps plan resources according to seasonal demand.
    """)

# GRAPH 3
c5, c6 = st.columns([2,1])
with c5:
    st.subheader("Age Distribution")
    bins = st.slider("Number of intervals (bins)", 5, 50, 20, key="age_bins")
    fig_age = px.histogram(df_selection, x="edad", nbins=bins, color_discrete_sequence=['#FFA200'])
    fig_age.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_age)
with c6:
    st.subheader("Graph Analysis")
    st.write("""
    - The graph shows the age concentration of users who report.
    - Most people are concentrated between 30 and 50 years old.
    - There are fewer cases among very young and very old ages.
    - The strongest core is in middle ages, extremes are rare.
    """)

# GRAPH 4
c7, c8 = st.columns([2,1])
with c7:
    if 'estado_civil' in df_selection.columns:
        st.subheader("Frequency by marital status")
        count_ms = df_selection['estado_civil'].value_counts().reset_index()
        fig_ms = px.bar(count_ms, x='index', y='estado_civil', color_discrete_sequence=["#9333EA"])
        fig_ms.update_layout(width=WIDTH, height=HEIGHT)
        st.plotly_chart(fig_ms)
with c8:
    st.subheader("Graph Analysis")
    st.write("""
    - The graph shows distribution by marital status of women who report.
    - Most are single women with around 250,000 records.
    - Followed by married women with approximately 150,000 cases.
    - Those in common-law relationships have about 50,000 registered cases.
    """)

# GRAPH 5
c9, c10 = st.columns([2,1])
with c9:
    st.subheader("Monthly call evolution")
    df_temp = df_selection.copy()
    df_temp['fecha_alta'] = pd.to_datetime(df_temp['fecha_alta'], errors='coerce')
    df_temp = df_temp.dropna(subset=['fecha_alta'])
    df_temp['year_month'] = df_temp['fecha_alta'].dt.to_period('M').astype(str)
    calls_per_month = df_temp.groupby('year_month').size().reset_index(name='total')
    fig_ev = px.line(calls_per_month, x='year_month', y='total', markers=True)
    fig_ev.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_ev)
with c10:
    st.subheader("Graph Analysis")
    st.write("""
    - The graph shows call evolution over time.
    - An initial increase is observed, then they remained with ups and downs.
    - Finally, a significant decrease is seen in recent periods.
    - Helps identify trends and evaluate intervention impact.
    """)

# TOPIC ANALYSIS
st.header("Topic Analysis")
topic_cols = ['tematica_1', 'tematica_2', 'tematica_3', 'tematica_4', 'tematica_5', 'tematica_6', 'tematica_7']
existing_topics = [c for c in topic_cols if c in df_selection.columns]
if existing_topics:
    df_exploded = df_selection.melt(value_vars=existing_topics, value_name='topic').dropna()
    top_15 = df_exploded['topic'].value_counts().head(15)
    fig_topics = px.bar(x=top_15.values, y=top_15.index, orientation='h', color_continuous_scale='Purples_r')
    fig_topics.update_layout(width=WIDTH, height=HEIGHT, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_topics)

# ==================== SURVEY AND CONTACT ====================
st.markdown("---")
st.header("Survey")
st.title("Tell us what happened that day")
st.markdown("Answer as honestly as possible")

with st.form("survey_form"):
    age_q = st.selectbox("What is your age?", ["Under 10", "10-15", "15-25", "25-35", "35-45", "Over 45"])
    sit_q = st.selectbox("Have you experienced any situation?", ["Sexual abuse", "Family violence", "Breach of trust", "Rape at school or work", "Other"])
    freq_q = st.selectbox("How often does it happen?", ["It happened once", "Occasionally", "Frequently", "It's happening to me now"])
    rel_q = st.selectbox("Relationship with the person", ["Partner", "Family member", "Work", "Other"])
    talk_q = st.selectbox("Have you talked to someone?", ["Yes", "No"])
    if st.form_submit_button("Submit"):
        if save_response({'age_group':age_q, 'situation':sit_q, 'frequency':freq_q, 'relationship':rel_q, 'talked_to_someone':talk_q}):
            st.success("Thank you for your trust! Your response has been saved.")
            st.balloons()

if st.button("Need help"):
    st.warning("""Call: 800 10 84 053 or 079. Remember, you are not alone. You can go to the following locations, don't be afraid to speak:
Women's Secretariat
Prolongación Corregidora Sur 210, 76074 Querétaro
442 215 3404          
Women's Secretariat of Querétaro Municipality
Galaxia 543, 76085 Santiago de Querétaro, Querétaro
442 238 7700
Municipal Women's Secretariat Corregidora Querétaro
Calle Monterrey, 76902 Corregidora, Querétaro""")

df_res = load_questionnaire_responses()
if not df_res.empty:
    st.sidebar.markdown("---")
    st.sidebar.metric("Responses received", len(df_res))
