import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sqlite3
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import os

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
    """Initializes the SQLite database"""
    try:
        conn = sqlite3.connect('questionnaire_women.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS questionnaire_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            age_group TEXT,
            situation TEXT,
            frequency TEXT,
            relationship TEXT,
            talked_to_someone TEXT
        )''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database init info: {e}")

def save_response(data):
    """Saves questionnaire responses to the database"""
    try:
        conn = sqlite3.connect('questionnaire_women.db')
        c = conn.cursor()
        c.execute('''INSERT INTO questionnaire_responses 
                    (timestamp, age_group, situation, frequency, relationship, talked_to_someone)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                  (datetime.now(), 
                   data['age_group'],
                   data['situation'],
                   data['frequency'],
                   data['relationship'],
                   data['talked_to_someone']))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving to database (Note: Cloud hosting may have restricted write access): {e}")
        return False

def load_questionnaire_responses():
    """Loads questionnaire responses for analysis"""
    try:
        conn = sqlite3.connect('questionnaire_women.db')
        df = pd.read_sql_query("SELECT * FROM questionnaire_responses", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

init_database()

# ==================== DATA LOADING (ZIP SUPPORT) ====================
@st.cache_data
def load_data():
    # Intentamos cargar el ZIP primero, si no, el CSV normal
    file_zip = "linea-mujeres-cdmx.zip"
    file_csv = "linea-mujeres-cdmx.csv"
    
    try:
        if os.path.exists(file_zip):
            df = pd.read_csv(file_zip, encoding="latin1", compression='zip')
        else:
            df = pd.read_csv(file_csv, encoding="latin1")
            
        df.columns = df.columns.str.lower().str.strip()
        return df
    except Exception as e:
        st.error(f"Critical Error: Could not find or read the data file. {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Please make sure 'linea-mujeres-cdmx.zip' or 'linea-mujeres-cdmx.csv' is in your GitHub repository.")
    st.stop()

# ==================== FILTERS ====================
st.sidebar.header("Filters")

available_states = df['estado_usuaria'].unique()
state = st.sidebar.multiselect(
    "Select State:",
    options=available_states,
    default=available_states[:3] if len(available_states) > 3 else available_states
)

if state:
    filtered_df_state = df[df['estado_usuaria'].isin(state)]
else:
    filtered_df_state = df

available_municipalities = filtered_df_state['municipio_usuaria'].unique()
municipality = st.sidebar.multiselect(
    "Select Municipality:",
    options=available_municipalities,
    default=available_municipalities[:5] if len(available_municipalities) > 5 else available_municipalities
)

if municipality:
    df_selection = filtered_df_state[filtered_df_state['municipio_usuaria'].isin(municipality)]
else:
    df_selection = filtered_df_state

col1, col2, col3 = st.columns(3)
col1.metric("Total Reports", f"{len(df_selection):,}")
avg_age = pd.to_numeric(df_selection['edad'], errors='coerce').mean()
col2.metric("Average Age", f"{avg_age:.0f}" if not pd.isna(avg_age) else "N/A")
col3.metric("Municipalities", f"{len(municipality) if municipality else len(df_selection['municipio_usuaria'].unique())}")

# ==================== GRAPHS ====================

# GRAPH 1: OCCUPATION
c1, c2 = st.columns([2,1])
with c1:
    st.subheader("Distribution by Occupation")
    fig_occupation = px.pie(df_selection, names='ocupacion', hole=0.6,
        color_discrete_sequence=["#E6CCFF","#D8B4FE","#C084FC","#A855F7","#9333EA","#7E22CE"])
    fig_occupation.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_occupation, use_container_width=False)
with c2:
    st.subheader("Graph Analysis")
    st.write("- Identification of occupational groups with highest presence in reports.")

# GRAPH 2: MONTHS
c3, c4 = st.columns([2,1])
with c3:
    st.subheader("Attentions by Month")
    month_counts = df_selection['mes_alta'].value_counts().reset_index()
    month_counts.columns = ['month', 'total']
    fig_month = px.bar(month_counts.sort_values(by='month'), x='month', y='total',
        labels={'month': 'Month', 'total': 'Calls'}, color_discrete_sequence=['#9333EA'])
    fig_month.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_month, use_container_width=False)
with c4:
    st.subheader("Graph Analysis")
    st.write("- Shows peaks indicating times of greatest demand.")

# GRAPH 3: AGE
c5, c6 = st.columns([2,1])
with c5:
    st.subheader("Age Distribution")
    bins = st.slider("Number of intervals (bins)", 5, 50, 20, key="age_bins")
    fig_age = px.histogram(df_selection, x="edad", nbins=bins, color_discrete_sequence=['#FFA200'])
    fig_age.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_age, use_container_width=False)
with c6:
    st.subheader("Graph Analysis")
    st.write("- Most reports concentrate between 30 and 50 years old.")

# GRAPH 4: MARITAL STATUS
c7, c8 = st.columns([2,1])
with c7:
    if 'estado_civil' in df.columns:
        st.subheader("Frequency by marital status")
        count_ms = df_selection['estado_civil'].value_counts().reset_index()
        count_ms.columns = ['marital_status', 'total']
        fig_ms = px.bar(count_ms, x='marital_status', y='total', color_discrete_sequence=["#9333EA"])
        fig_ms.update_layout(width=WIDTH, height=HEIGHT, xaxis_tickangle=45)
        st.plotly_chart(fig_ms, use_container_width=False)
with c8:
    st.subheader("Graph Analysis")
    st.write("- Single and married women represent the majority of records.")

# GRAPH 5: EVOLUTION
c9, c10 = st.columns([2,1])
with c9:
    st.subheader("Monthly call evolution")
    df_temp = df_selection.copy()
    df_temp['fecha_alta'] = pd.to_datetime(df_temp['fecha_alta'], errors='coerce')
    df_temp = df_temp.dropna(subset=['fecha_alta'])
    df_temp['year_month'] = df_temp['fecha_alta'].dt.to_period('M').astype(str)
    calls_per_month = df_temp.groupby('year_month').size().reset_index()
    calls_per_month.columns = ['year_month', 'total']
    fig_ev = px.line(calls_per_month, x='year_month', y='total', markers=True)
    fig_ev.update_layout(width=WIDTH, height=HEIGHT, xaxis_tickangle=90)
    st.plotly_chart(fig_ev, use_container_width=False)
with c10:
    st.subheader("Graph Analysis")
    st.write("- Shows trends over time and impact evaluation.")

# GRAPH 6: CLUSTERS
c11, c12 = st.columns([2,1])
with c11:
    st.subheader("Call clusters (Age vs Service)")
    df_num = df_selection.select_dtypes(include=['int64','float64']).fillna(df_selection.median(numeric_only=True))
    if not df_num.empty and len(df_num) > 3:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_num)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_selection['cluster'] = kmeans.fit_predict(scaled_data)
        if 'edad' in df_selection.columns and 'servicio' in df_selection.columns:
            fig_cl = px.scatter(df_selection, x='edad', y='servicio', color='cluster', color_continuous_scale='viridis')
            fig_cl.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig_cl, use_container_width=False)
with c12:
    st.subheader("Graph Analysis")
    st.write("- Identifies patterns according to age ranges and service type.")

# ==================== TOPIC ANALYSIS ====================
st.header("Topic Analysis")
topic_columns = ['tematica_1', 'tematica_2', 'tematica_3', 'tematica_4', 'tematica_5', 'tematica_6', 'tematica_7']
existing_topics = [col for col in topic_columns if col in df.columns]

if existing_topics:
    df_temp_top = df_selection.copy()
    for col in existing_topics:
        df_temp_top[col] = df_temp_top[col].fillna('Not specified')
    
    df_temp_top['topics_list'] = df_temp_top[existing_topics].apply(lambda x: x.tolist(), axis=1)
    df_exploded = df_temp_top.explode('topics_list')
    df_exploded = df_exploded[df_exploded['topics_list'] != 'Not specified']
    df_exploded = df_exploded.rename(columns={'topics_list': 'topic'})
    
    df_exploded['edad'] = pd.to_numeric(df_exploded['edad'], errors='coerce')
    df_exploded['age_group'] = pd.cut(
        df_exploded['edad'],
        bins=[0, 18, 25, 35, 45, 100],
        labels=['<18 years', '18-25 years', '26-35 years', '36-45 years', '>45 years']
    )
    
    # TOP 15 TOPICS
    c_topic1, c_topic2 = st.columns([2,1])
    with c_topic1:
        st.subheader("Top 15 most reported topics")
        top_topics = df_exploded['topic'].value_counts().head(15)
        fig_top_topics = px.bar(x=top_topics.values, y=top_topics.index, orientation='h',
            color=top_topics.values, color_continuous_scale='Purples_r')
        fig_top_topics.update_layout(width=WIDTH, height=HEIGHT, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top_topics, use_container_width=False)
    with c_topic2:
        st.write("- Darker bars indicate priority problems.")

# ==================== SURVEY ====================
st.markdown("---")
st.header("Survey")
with st.form(key="questionnaire_form"):
    age_group = st.selectbox("Age?", ["Under 10", "10-15", "15-25", "25-35", "35-45", "Over 45"])
    situation = st.selectbox("Situation?", ["Sexual abuse", "Family violence", "Breach of trust", "Rape", "Other"])
    frequency = st.selectbox("Frequency?", ["Once", "Occasionally", "Frequently", "Now"])
    relationship = st.selectbox("Relationship?", ["Partner", "Family", "Work", "Other"])
    talked_to_someone = st.selectbox("Talked to someone?", ["Yes", "No"])
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        response_data = {'age_group': age_group, 'situation': situation, 'frequency': frequency, 
                         'relationship': relationship, 'talked_to_someone': talked_to_someone}
        if save_response(response_data):
            st.success("Thank you for your trust!")
            st.balloons()

if st.button("Need help"):
    st.warning("Call: 800 10 84 053. You are not alone.")
