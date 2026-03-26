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
        st.error(f"Error saving: {e}")
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

# Initialize database
init_database()
# ==================== END DATABASE CONFIGURATION ====================
@st.cache_data
def load_data():
    csv_filename = "linea-mujeres-cdmx.csv"
    zip_filename = "linea-mujeres-cdmx.zip"
    
    # Check if CSV already exists
    if os.path.exists(csv_filename):
        try:
            df = pd.read_csv(csv_filename, encoding="latin1")
            df.columns = df.columns.str.lower().str.strip()
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return pd.DataFrame()
    
    # Try to extract from ZIP
    if os.path.exists(zip_filename):
        try:
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall()
            
            # Now try to read the CSV
            if os.path.exists(csv_filename):
                df = pd.read_csv(csv_filename, encoding="latin1")
                df.columns = df.columns.str.lower().str.strip()
                return df
            else:
                st.error(f"CSV file '{csv_filename}' not found in ZIP archive")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error extracting or reading ZIP file: {e}")
            return pd.DataFrame()
    else:
        st.error(f"Neither '{csv_filename}' nor '{zip_filename}' found in the repository")
        return pd.DataFrame()

# Load the data
df = load_data()

# Check if data loaded successfully
if df.empty:
    st.error("Failed to load data. Please check that the data files are present.")
    st.stop()

# ==================== FILTERS ====================
st.sidebar.header("Filters")

# Filter by State
available_states = df['estado_usuaria'].unique()
# ==================== FILTERS ====================
st.sidebar.header("Filters")

# Filter by State
available_states = df['estado_usuaria'].unique()
state = st.sidebar.multiselect(
    "Select State:",
    options=available_states,
    default=available_states[:3] if len(available_states) > 3 else available_states
)

# Filter by state first
if state:
    filtered_df_state = df[df['estado_usuaria'].isin(state)]
else:
    filtered_df_state = df

# Filter by Municipality
available_municipalities = filtered_df_state['municipio_usuaria'].unique()
municipality = st.sidebar.multiselect(
    "Select Municipality:",
    options=available_municipalities,
    default=available_municipalities[:5] if len(available_municipalities) > 5 else available_municipalities
)

# Apply municipality filter
if municipality:
    df_selection = filtered_df_state[filtered_df_state['municipio_usuaria'].isin(municipality)]
else:
    df_selection = filtered_df_state

col1, col2, col3 = st.columns(3)
col1.metric("Total Reports", f"{len(df_selection):,}")
avg_age = pd.to_numeric(df_selection['edad'], errors='coerce').mean()
col2.metric("Average Age", f"{avg_age:.0f}" if not pd.isna(avg_age) else "N/A")
col3.metric("Municipalities", f"{len(municipality) if municipality else len(df_selection['municipio_usuaria'].unique())}")

# ==================== GRAPH 1: DISTRIBUTION BY OCCUPATION ====================
c1, c2 = st.columns([2,1])

with c1:
    st.subheader("Distribution by Occupation")
    fig_occupation = px.pie(df_selection, names='ocupacion', hole=0.6,
        color_discrete_sequence=["#E6CCFF","#D8B4FE","#C084FC","#A855F7","#9333EA","#7E22CE"])
    fig_occupation.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_occupation, use_container_width=False)

with c2:
    st.subheader("Graph Analysis")
    st.write("""
    - The graph shows the distribution of occupations of users who made calls.
    - It shows which occupational groups have the highest presence in reports.
    - Larger portions indicate labor sectors with more cases.
    - This helps focus prevention campaigns on specific sectors.
    """)

# ==================== GRAPH 2: ATTENTIONS BY MONTH ====================
c3, c4 = st.columns([2,1])

with c3:
    st.subheader("Attentions by Month")
    month_counts = df_selection['mes_alta'].value_counts().reset_index()
    month_counts.columns = ['month', 'total']
    fig_month = px.bar(month_counts.sort_values(by='month'), x='month', y='total',
        labels={'month': 'Month of the Year', 'total': 'Number of calls'},
        color_discrete_sequence=['#9333EA'])
    fig_month.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_month, use_container_width=False)

with c4:
    st.subheader("Graph Analysis")
    st.write("""
    - The graph shows call distribution for each month of the year.
    - Identifies months with the highest and lowest number of reports.
    - Highest peaks indicate times of greatest demand for attention.
    - Helps plan resources according to seasonal demand.
    """)

# ==================== GRAPH 3: AGE DISTRIBUTION ====================
c5, c6 = st.columns([2,1])

with c5:
    st.subheader("Age Distribution")
    bins = st.slider("Number of intervals (bins)", 5, 50, 20, key="age_bins")
    fig_age = px.histogram(df_selection, x="edad", nbins=bins,
        title="Age Distribution of Users", color_discrete_sequence=['#FFA200'])
    fig_age.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig_age, use_container_width=False)

with c6:
    st.subheader("Graph Analysis")
    st.write("""
    - The graph shows the age concentration of users who report.
    - Most people are concentrated between 30 and 50 years old.
    - There are fewer cases among very young and very old ages.
    - The strongest core is in middle ages, extremes are rare.
    """)

# ==================== GRAPH 4: FREQUENCY BY MARITAL STATUS ====================
c7, c8 = st.columns([2,1])

with c7:
    if 'estado_civil' in df.columns:
        st.subheader("Frequency by marital status")
        count_ms = df['estado_civil'].value_counts().reset_index()
        count_ms.columns = ['marital_status', 'total']
        fig_ms = px.bar(count_ms, x='marital_status', y='total', color_discrete_sequence=["#9333EA"])
        fig_ms.update_layout(width=WIDTH, height=HEIGHT, xaxis_tickangle=45)
        st.plotly_chart(fig_ms, use_container_width=False)

with c8:
    st.subheader("Graph Analysis")
    st.write("""
    - The graph shows distribution by marital status of women who report.
    - Most are single women with around 250,000 records.
    - Followed by married women with approximately 150,000 cases.
    - Those in common-law relationships have about 50,000 registered cases.
    """)

# ==================== GRAPH 5: MONTHLY CALL EVOLUTION ====================
c9, c10 = st.columns([2,1])

with c9:
    st.subheader("Monthly call evolution")
    df_temp = df.copy()
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
    st.write("""
    - The graph shows call evolution over time.
    - An initial increase is observed, then they remained with ups and downs.
    - Finally, a significant decrease is seen in recent periods.
    - Helps identify trends and evaluate intervention impact.
    """)

# ==================== GRAPH 6: CLUSTERS AGE VS SERVICE ====================
c11, c12 = st.columns([2,1])

with c11:
    st.subheader("Call clusters (Age vs Service)")
    df_num = df.select_dtypes(include=['int64','float64']).fillna(df.median(numeric_only=True))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_num)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    if 'edad' in df.columns and 'servicio' in df.columns:
        fig_cl = px.scatter(df, x='edad', y='servicio', color='cluster', color_continuous_scale='viridis')
        fig_cl.update_layout(width=WIDTH, height=HEIGHT)
        st.plotly_chart(fig_cl, use_container_width=False)

with c12:
    st.subheader("Graph Analysis")
    st.write("""
    - The horizontal axis indicates the age of women who called.
    - The vertical axis indicates the type of counseling required.
    - Points represent calls and color indicates which age group they belong to.
    - Helps identify service patterns according to age ranges.
    """)

# ==================== TOPIC ANALYSIS ====================
st.header("Topic Analysis")

# Prepare topic data
topic_columns = ['tematica_1', 'tematica_2', 'tematica_3', 'tematica_4', 'tematica_5', 'tematica_6', 'tematica_7']
existing_topics = [col for col in topic_columns if col in df.columns]

if existing_topics:
    # Create expanded version for analysis
    df_temp = df_selection.copy()
    for col in existing_topics:
        df_temp[col] = df_temp[col].fillna('Not specified')
    
    df_temp['topics_list'] = df_temp[existing_topics].apply(lambda x: x.tolist(), axis=1)
    df_exploded = df_temp.explode('topics_list')
    df_exploded = df_exploded[df_exploded['topics_list'] != 'Not specified']
    df_exploded = df_exploded.rename(columns={'topics_list': 'topic'})
    
    # Create age groups
    df_exploded['edad'] = pd.to_numeric(df_exploded['edad'], errors='coerce')
    df_exploded['age_group'] = pd.cut(
        df_exploded['edad'],
        bins=[0, 18, 25, 35, 45, 100],
        labels=['<18 years', '18-25 years', '26-35 years', '36-45 years', '>45 years']
    )
    
    # GRAPH 7: TOP 15 MOST REPORTED TOPICS
    c_topic1, c_topic2 = st.columns([2,1])
    
    with c_topic1:
        st.subheader("Top 15 most reported topics")
        
        top_topics = df_exploded['topic'].value_counts().head(15)
        
        fig_top_topics = px.bar(
            x=top_topics.values, 
            y=top_topics.index,
            orientation='h',
            labels={'x': 'Number of cases', 'y': 'Topic'},
            color=top_topics.values,
            color_continuous_scale='Purples_r'
        )
        fig_top_topics.update_layout(width=WIDTH, height=HEIGHT, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top_topics, use_container_width=False)
    
    with c_topic2:
        st.subheader("Graph Analysis")
        st.write("""
        - The graph shows the 15 most frequent problems reported by users.
        - Longer bars with darker purple color represent the most common problems.
        - A clear concentration is observed in the first 3-4 topics, indicating priority problems.
        - The topic with the highest number of cases should be the main focus of support programs.
        """)
    
    # GRAPH 8: TOPIC DISTRIBUTION BY AGE GROUP
    c_topic3, c_topic4 = st.columns([2,1])
    
    with c_topic3:
        st.subheader("Topic distribution by age group")
        
        # Select top 8 topics
        top8 = df_exploded['topic'].value_counts().head(8).index
        df_top8 = df_exploded[df_exploded['topic'].isin(top8)]
        
        # Create contingency table
        age_topic = pd.crosstab(df_top8['topic'], df_top8['age_group'])
        
        # Purple colors for age groups
        purple_colors = ['#4A0E4E', '#6B2E6B', '#8B4B8B', '#AA6EAA', '#C999C9']
        
        fig_topic_age = px.bar(
            age_topic,
            labels={'value': 'Number of cases', 'topic': 'Topic', 'variable': 'Age Group'},
            color_discrete_sequence=purple_colors,
            barmode='stack'
        )
        fig_topic_age.update_layout(width=WIDTH, height=HEIGHT, xaxis_tickangle=45)
        st.plotly_chart(fig_topic_age, use_container_width=False)
    
    with c_topic4:
        st.subheader("Graph Analysis")
        st.write("""
        - The graph shows how problems are distributed according to user age.
        - The darkest purple color (#4A0E4E) represents the youngest group (<18 years).
        - The lightest purple color (#C999C9) represents the oldest group (>45 years).
        - Shows which problems affect younger women more and which affect older women.
        - Balanced colors indicate problems affecting all ages.
        """)

else:
    st.warning("No topic columns found in the data")

# ==================== EDUCATION LEVEL ANALYSIS ====================
st.header("Education Level Analysis")

if 'escolaridad' in df.columns:
    # Prepare topic data for education analysis
    topic_columns = ['tematica_1', 'tematica_2', 'tematica_3', 'tematica_4', 'tematica_5', 'tematica_6', 'tematica_7']
    existing_topics_edu = [col for col in topic_columns if col in df.columns]
    
    if existing_topics_edu:
        df_temp_edu = df_selection.copy()
        for col in existing_topics_edu:
            df_temp_edu[col] = df_temp_edu[col].fillna('Not specified')
        
        df_temp_edu['topics_list'] = df_temp_edu[existing_topics_edu].apply(lambda x: x.tolist(), axis=1)
        df_exploded_edu = df_temp_edu.explode('topics_list')
        df_exploded_edu = df_exploded_edu[df_exploded_edu['topics_list'] != 'Not specified']
        df_exploded_edu = df_exploded_edu.rename(columns={'topics_list': 'topic'})
    else:
        df_exploded_edu = df_selection.copy()
    
    # GRAPH 9: EDUCATION LEVEL VS TYPE OF VIOLENCE
    c_edu1, c_edu2 = st.columns([2,1])
    
    with c_edu1:
        st.subheader("Education Level vs Type of Violence")
        
        if 'topic' in df_exploded_edu.columns:
            # Create contingency table
            education_violence = pd.crosstab(df_exploded_edu['topic'], df_exploded_edu['escolaridad'])
            
            # Select top 10 topics
            top10_edu = education_violence.sum(axis=1).sort_values(ascending=False).head(10).index
            education_violence_top = education_violence.loc[top10_edu]
            
            # Heatmap
            fig_edu_viol = px.imshow(
                education_violence_top,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Purples',
                title="Relationship: Education Level vs Type of Violence",
                labels={'x': 'Education Level', 'y': 'Type of Violence', 'color': 'Number of cases'}
            )
            fig_edu_viol.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig_edu_viol, use_container_width=False)
        else:
            st.info("No topic data available for this analysis")
    
    with c_edu2:
        st.subheader("Graph Analysis")
        st.write("""
        - The graph shows which types of violence are most common according to education level.
        - Darker colors indicate higher concentration of cases in that combination.
        - Helps identify if psychological violence is more reported by women with higher education.
        - Helps design prevention campaigns adapted to each education level.
        """)
    
    # GRAPH 10: MARITAL STATUS BY EDUCATION LEVEL
    c_edu3, c_edu4 = st.columns([2,1])
    
    with c_edu3:
        st.subheader("Marital status by education level")
        
        if 'estado_civil' in df_exploded_edu.columns:
            education_marital = pd.crosstab(df_exploded_edu['escolaridad'], df_exploded_edu['estado_civil'])
            
            fig_edu_marital = px.bar(
                education_marital,
                barmode='stack',
                title="Marital Status by Education Level",
                labels={'value': 'Number of cases', 'escolaridad': 'Education Level', 'variable': 'Marital Status'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_edu_marital.update_layout(width=WIDTH, height=HEIGHT, xaxis_tickangle=45)
            st.plotly_chart(fig_edu_marital, use_container_width=False)
        else:
            st.info("No marital status data available for this analysis")
    
    with c_edu4:
        st.subheader("Graph Analysis")
        st.write("""
        - The graph shows marital status distribution according to education level.
        - Helps identify if women with higher education have different marital status patterns.
        - Shows if there's a higher proportion of single women in higher education levels.
        - Helps understand the socio-family context according to education level.
        """)

else:
    st.warning("Column 'escolaridad' not found in the data")
    st.info("To add this analysis, make sure your CSV file contains the 'escolaridad' column")

# ==================== QUESTIONNAIRE ====================
st.markdown("---")
st.header("Survey")
st.title("Tell us what happened that day")
st.markdown("Answer as honestly as possible")

with st.form(key="questionnaire_form"):
    age_group = st.selectbox(
        "What is your age?",
        ["Under 10", "10-15", "15-25", "25-35", "35-45", "Over 45"]
    )
    
    situation = st.selectbox(
        "Have you experienced any situation?",
        ["Sexual abuse", "Family violence", "Breach of trust", "Rape at school or work", "Other"]
    )
    
    frequency = st.selectbox(
        "How often does it happen?",
        ["It happened once", "Occasionally", "Frequently", "It's happening to me now"]
    )
    
    relationship = st.selectbox(
        "Relationship with the person",
        ["Partner", "Family member", "Work", "Other"]
    )
    
    talked_to_someone = st.selectbox(
        "Have you talked to someone?",
        ["Yes", "No"]
    )
    
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        response_data = {
            'age_group': age_group,
            'situation': situation,
            'frequency': frequency,
            'relationship': relationship,
            'talked_to_someone': talked_to_someone
        }
        
        if save_response(response_data):
            st.success("Thank you for your trust! Your response has been saved.")
            st.balloons()
        else:
            st.error("There was an error saving your response. Please try again.")

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

df_responses = load_questionnaire_responses()
if not df_responses.empty:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Questionnaire Statistics")
    st.sidebar.metric("Responses received", len(df_responses))
    st.sidebar.metric("Last response", df_responses['timestamp'].max().split()[0] if not df_responses.empty else "N/A")
