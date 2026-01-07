"""
Life Expectancy Prediction Dashboard
=====================================
A Streamlit application that predicts life expectancy based on vaccine coverage
and provides comprehensive data visualizations.

Features:
- Loads pre-trained models from models/ directory
- Interactive charts and visualizations
- Data upload and management capabilities
- Dataset deletion feature
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="VaccineLife | Life Expectancy Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================
MODELS_DIR = "models"
DEFAULT_DATA_PATH = "data/joined_cty_vacc.txt"

# Model artifact paths
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(MODELS_DIR, "imputer.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e4a 0%, #0d0d2b 100%);
        border-right: 1px solid #3d3d8f;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a4e 0%, #2d2d6e 100%);
        border: 1px solid #4d4d9f;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
    }
    
    [data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0ff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0088ff 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Delete button styling */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.3);
    }
    
    .stButton > button[kind="secondary"]:hover {
        box-shadow: 0 6px 20px rgba(255, 68, 68, 0.5);
    }
    
    /* Info boxes */
    .stAlert {
        background: linear-gradient(135deg, #1a3a4e 0%, #0d2a3b 100%);
        border: 1px solid #00d4ff;
        border-radius: 10px;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(135deg, #004d40 0%, #00695c 100%);
        border: 2px solid #00ff88;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.2);
    }
    
    .prediction-value {
        font-size: 4rem;
        font-weight: bold;
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a4e;
        border-radius: 8px 8px 0 0;
        border: 1px solid #4d4d9f;
        color: #a0a0ff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0088ff 100%);
        color: white !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        margin: 30px 0;
    }
    
    /* Data management section */
    .data-management {
        background: linear-gradient(135deg, #2a2a5e 0%, #1a1a4e 100%);
        border: 1px solid #4d4d9f;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# VACCINE INFORMATION
# ============================================================================
VACCINE_INFO = {
    'BCG': 'Bacillus Calmette-Guérin (Tuberculosis)',
    'DTP1': 'Diphtheria-Tetanus-Pertussis (1st dose)',
    'DTP3': 'Diphtheria-Tetanus-Pertussis (3rd dose)',
    'HEPB3': 'Hepatitis B (3rd dose)',
    'HEPBB': 'Hepatitis B (birth dose)',
    'HIB3': 'Haemophilus influenzae type b (3rd dose)',
    'IPV1': 'Inactivated Polio Vaccine (1st dose)',
    'IPV2': 'Inactivated Polio Vaccine (2nd dose)',
    'MCV1': 'Measles-containing Vaccine (1st dose)',
    'MCV2': 'Measles-containing Vaccine (2nd dose)',
    'MENGA': 'Meningococcal A conjugate vaccine',
    'PCV3': 'Pneumococcal Conjugate Vaccine (3rd dose)',
    'POL3': 'Polio (3rd dose)',
    'RCV1': 'Rubella-containing Vaccine (1st dose)',
    'ROTAC': 'Rotavirus (completed series)',
    'YFV': 'Yellow Fever Vaccine'
}

VACCINE_COLS = list(VACCINE_INFO.keys())

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'imputer' not in st.session_state:
    st.session_state.imputer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None

# ============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# ============================================================================
def load_data_from_file(file_path):
    """Load data from a file path"""
    try:
        if file_path.endswith('.csv') or file_path.endswith('.txt'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {str(e)}")
        return None


def load_data_from_upload(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, TXT, or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def preprocess_data(df):
    """Preprocess the dataset for visualization"""
    df_processed = df.copy()
    
    for col in VACCINE_COLS:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    if 'life_expectancy' in df_processed.columns:
        df_processed['life_expectancy'] = pd.to_numeric(df_processed['life_expectancy'], errors='coerce')
    
    return df_processed


def clear_uploaded_data():
    """Clear the uploaded dataset from session state"""
    st.session_state.df = None
    st.session_state.data_loaded = False
    st.session_state.data_source = None


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================
def load_model_artifacts():
    """Load model artifacts from the models directory"""
    try:
        # Check if models directory exists
        if not os.path.exists(MODELS_DIR):
            st.error(f"Models directory '{MODELS_DIR}' not found. Please create it and add your model files.")
            return False
        
        # Load model (required)
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at '{MODEL_PATH}'. Please add your trained model.")
            return False
        
        st.session_state.model = joblib.load(MODEL_PATH)
        
        # Load optional artifacts
        if os.path.exists(SCALER_PATH):
            st.session_state.scaler = joblib.load(SCALER_PATH)
        
        if os.path.exists(IMPUTER_PATH):
            st.session_state.imputer = joblib.load(IMPUTER_PATH)
        
        if os.path.exists(FEATURE_NAMES_PATH):
            st.session_state.feature_names = joblib.load(FEATURE_NAMES_PATH)
        else:
            st.session_state.feature_names = VACCINE_COLS
        
        st.session_state.model_loaded = True
        return True
        
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        return False

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def prepare_features(vaccine_data):
    """Prepare feature vector for prediction"""
    feature_names = st.session_state.feature_names or VACCINE_COLS
    features = []
    
    for vaccine in feature_names:
        features.append(vaccine_data.get(vaccine, np.nan))
    
    features_array = np.array(features).reshape(1, -1)
    
    if st.session_state.imputer:
        features_array = st.session_state.imputer.transform(features_array)
    
    if st.session_state.scaler:
        features_array = st.session_state.scaler.transform(features_array)
    
    return features_array


def make_prediction(vaccine_data):
    """Make life expectancy prediction"""
    if not st.session_state.model_loaded:
        st.error("Model not loaded. Please ensure model files are in the models/ directory.")
        return None
    
    try:
        features = prepare_features(vaccine_data)
        prediction = st.session_state.model.predict(features)[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_data_summary(df):
    """Create data summary section"""
    st.subheader(" Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        if 'country' in df.columns:
            st.metric("Countries", df['country'].nunique())
    with col3:
        if 'year' in df.columns:
            st.metric("Years Covered", f"{df['year'].min()}-{df['year'].max()}")
    with col4:
        if 'life_expectancy' in df.columns:
            st.metric("Avg Life Expectancy", f"{df['life_expectancy'].mean():.1f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Missing Data Analysis")
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df.head(10), x='Missing %', y='Column', 
                        orientation='h',
                        title='Top 10 Columns with Missing Data',
                        color='Missing %',
                        color_continuous_scale='Reds')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing data found!")


def create_global_overview(df):
    """Create global overview visualizations"""
    st.subheader("Global Life Expectancy Trends")
    
    if 'life_expectancy' not in df.columns or 'year' not in df.columns:
        st.warning("Required columns not found for this visualization.")
        return
    
    yearly_avg = df.groupby('year')['life_expectancy'].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_avg['year'],
        y=yearly_avg['life_expectancy'],
        mode='lines+markers',
        name='Global Average',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Global Average Life Expectancy Over Time',
        xaxis_title='Year',
        yaxis_title='Life Expectancy (years)',
        hovermode='x unified',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a0a0ff')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    if 'country' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Countries (Latest Year)")
            latest_year = df['year'].max()
            latest_data = df[df['year'] == latest_year].nlargest(10, 'life_expectancy')
            
            fig = px.bar(latest_data, x='life_expectancy', y='country',
                        orientation='h',
                        title=f'Top 10 Countries by Life Expectancy ({latest_year})',
                        color='life_expectancy',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Bottom 10 Countries (Latest Year)")
            bottom_data = df[df['year'] == latest_year].nsmallest(10, 'life_expectancy')
            
            fig = px.bar(bottom_data, x='life_expectancy', y='country',
                        orientation='h',
                        title=f'Bottom 10 Countries by Life Expectancy ({latest_year})',
                        color='life_expectancy',
                        color_continuous_scale='Reds')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def create_vaccine_coverage_analysis(df):
    """Create vaccine coverage analysis"""
    st.subheader("💉 Vaccine Coverage Analysis")
    
    available_vaccines = [v for v in VACCINE_COLS if v in df.columns]
    
    if not available_vaccines:
        st.warning("No vaccine data found in the dataset.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Coverage by Vaccine")
        avg_coverage = df[available_vaccines].mean().sort_values(ascending=True)
        
        fig = go.Figure(go.Bar(
            x=avg_coverage.values,
            y=avg_coverage.index,
            orientation='h',
            marker=dict(
                color=avg_coverage.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Coverage %")
            ),
            text=avg_coverage.values.round(1),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Average Vaccine Coverage (%)',
            xaxis_title='Coverage Percentage',
            yaxis_title='Vaccine',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0a0ff')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Coverage Trends Over Time")
        selected_vaccines = st.multiselect(
            "Select vaccines to compare:",
            options=available_vaccines,
            default=available_vaccines[:3] if len(available_vaccines) >= 3 else available_vaccines
        )
        
        if selected_vaccines and 'year' in df.columns:
            fig = go.Figure()
            
            for vaccine in selected_vaccines:
                yearly_avg = df.groupby('year')[vaccine].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=yearly_avg['year'],
                    y=yearly_avg[vaccine],
                    mode='lines+markers',
                    name=vaccine,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title='Vaccine Coverage Trends',
                xaxis_title='Year',
                yaxis_title='Coverage %',
                height=500,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a0a0ff')
            )
            
            st.plotly_chart(fig, use_container_width=True)


def create_scatter_analysis(df):
    """Create scatter plot analysis"""
    st.subheader("Vaccine Coverage vs Life Expectancy")
    
    available_vaccines = [v for v in VACCINE_COLS if v in df.columns]
    
    if not available_vaccines or 'life_expectancy' not in df.columns:
        st.warning("Required data not available for scatter analysis.")
        return
    
    selected_vaccine = st.selectbox(
        "Select vaccine for analysis:",
        options=available_vaccines,
        index=0
    )
    
    if selected_vaccine:
        fig = px.scatter(
            df.dropna(subset=[selected_vaccine, 'life_expectancy']),
            x=selected_vaccine,
            y='life_expectancy',
            color='year' if 'year' in df.columns else None,
            hover_data=['country'] if 'country' in df.columns else None,
            title=f'{selected_vaccine} Coverage vs Life Expectancy',
            trendline='ols',
            opacity=0.6
        )
        
        fig.update_layout(
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0a0ff')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        corr = df[[selected_vaccine, 'life_expectancy']].corr().iloc[0, 1]
        st.info(f" Correlation coefficient: {corr:.3f}")


def create_correlation_analysis(df):
    """Create correlation analysis"""
    st.subheader(" Correlation Matrix")
    
    available_vaccines = [v for v in VACCINE_COLS if v in df.columns]
    
    if not available_vaccines or 'life_expectancy' not in df.columns:
        st.warning("Insufficient data for correlation analysis.")
        return
    
    cols_to_analyze = available_vaccines + ['life_expectancy']
    corr_matrix = df[cols_to_analyze].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu',
        aspect='auto',
        title='Correlation Heatmap: Vaccines and Life Expectancy'
    )
    
    fig.update_layout(
        height=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a0a0ff')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top correlations with life expectancy
    st.subheader("Top Correlations with Life Expectancy")
    life_corr = corr_matrix['life_expectancy'].drop('life_expectancy').sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Strongest Positive Correlations:**")
        for i, (vaccine, corr) in enumerate(life_corr.head(5).items(), 1):
            st.write(f"{i}. **{vaccine}**: {corr:.3f}")
    
    with col2:
        st.markdown("**Weakest Correlations:**")
        for i, (vaccine, corr) in enumerate(life_corr.tail(5).items(), 1):
            st.write(f"{i}. **{vaccine}**: {corr:.3f}")


def create_country_analysis(df):
    """Create country-specific analysis"""
    st.subheader("Country-Specific Analysis")
    
    if 'country' not in df.columns:
        st.warning("Country information not available in dataset.")
        return
    
    selected_country = st.selectbox(
        "Select a country:",
        options=sorted(df['country'].unique())
    )
    
    if selected_country:
        country_data = df[df['country'] == selected_country].sort_values('year')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{selected_country} - Life Expectancy Trend")
            
            if 'life_expectancy' in country_data.columns and 'year' in country_data.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=country_data['life_expectancy'],
                    mode='lines+markers',
                    name='Life Expectancy',
                    line=dict(color='#00ff88', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Life Expectancy (years)',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#a0a0ff')
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(f"💉 {selected_country} - Vaccine Coverage")
            
            available_vaccines = [v for v in VACCINE_COLS if v in country_data.columns]
            
            if available_vaccines and 'year' in country_data.columns:
                latest_year = country_data['year'].max()
                latest_data = country_data[country_data['year'] == latest_year][available_vaccines].iloc[0]
                latest_data = latest_data.dropna().sort_values(ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=latest_data.values,
                    y=latest_data.index,
                    orientation='h',
                    marker=dict(color='#00d4ff')
                ))
                
                fig.update_layout(
                    title=f'Vaccine Coverage in {latest_year}',
                    xaxis_title='Coverage %',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#a0a0ff')
                )
                
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PREDICTION INTERFACE
# ============================================================================
def create_prediction_interface():
    """Create the prediction interface"""
    st.subheader("Life Expectancy Prediction")
    
    if not st.session_state.model_loaded:
        st.warning("Model not loaded. Please ensure model files are in the models/ directory.")
        if st.button("Try Loading Model Again"):
            load_model_artifacts()
            st.rerun()
        return
    
    st.info("Enter vaccine coverage percentages (0-100) to predict life expectancy.")
    
    feature_names = st.session_state.feature_names or VACCINE_COLS
    
    vaccine_data = {}
    
    cols_per_row = 4
    for i in range(0, len(feature_names), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(feature_names):
                vaccine = feature_names[i + j]
                vaccine_label = VACCINE_INFO.get(vaccine, vaccine)
                with col:
                    vaccine_data[vaccine] = st.number_input(
                        f"{vaccine}",
                        min_value=0.0,
                        max_value=100.0,
                        value=50.0,
                        step=1.0,
                        help=vaccine_label,
                        key=f"vaccine_{vaccine}"
                    )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("Predict Life Expectancy", type="primary", use_container_width=True):
            prediction = make_prediction(vaccine_data)
            
            if prediction is not None:
                st.markdown(f"""
                <div class="prediction-result">
                    <h2 style="color: #00d4ff; margin-bottom: 10px;">Predicted Life Expectancy</h2>
                    <div class="prediction-value">{prediction:.1f}</div>
                    <p style="color: #a0ffcc; font-size: 1.2rem;">years</p>
                </div>
                """, unsafe_allow_html=True)
                
                if prediction >= 75:
                    st.success("High life expectancy - associated with well-developed healthcare systems.")
                elif prediction >= 65:
                    st.info(" Moderate life expectancy - improvements in coverage could increase this.")
                else:
                    st.warning("Lower life expectancy - significant improvements in coverage may help.")


# ============================================================================
# SIDEBAR
# ============================================================================
def create_sidebar():
    """Create the sidebar with navigation and file uploads"""
    
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #00d4ff; font-size: 1.8rem;"> VaccineLife</h1>
        <p style="color: #a0a0ff;">Life Expectancy Predictor</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Flow selection
    st.sidebar.subheader("Select Your Flow")
    flow = st.sidebar.radio(
        "What would you like to do?",
        options=["Visualization Only", "Prediction Only", "Both"],
        index=2,
        help="Choose whether you want to visualize data, make predictions, or both"
    )
    
    st.sidebar.markdown("---")
    
    # Data management section
    if flow in ["Visualization Only", "Both"]:
        st.sidebar.subheader("Data Management")
        
        # Show current data status
        if st.session_state.data_loaded:
            st.sidebar.markdown(
                f"""
                <div class="data-management">
                    <p style="color: #00ff88; font-weight: bold;">Data Loaded</p>
                    <p style="color: #a0a0ff; font-size: 0.9rem;">Source: {st.session_state.data_source}</p>
                    <p style="color: #a0a0ff; font-size: 0.9rem;">Records: {len(st.session_state.df):,}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Delete button
            if st.sidebar.button("Delete Current Dataset", type="secondary", use_container_width=True):
                clear_uploaded_data()
                st.sidebar.success("Dataset cleared!")
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Data upload section
        st.sidebar.subheader("Upload New Dataset")
        uploaded_data = st.sidebar.file_uploader(
            "Upload Dataset (CSV/TXT/Excel)",
            type=['csv', 'txt', 'xlsx', 'xls'],
            help="Upload your vaccine coverage dataset",
            key="data_uploader"
        )
        
        if uploaded_data is not None:
            df = load_data_from_upload(uploaded_data)
            if df is not None:
                st.session_state.df = preprocess_data(df)
                st.session_state.data_loaded = True
                st.session_state.data_source = uploaded_data.name
                st.sidebar.success(f"Data loaded: {len(df)} records")
                st.rerun()
        
        # Load default dataset button
        if not st.session_state.data_loaded:
            if os.path.exists(DEFAULT_DATA_PATH):
                if st.sidebar.button("Load Default Dataset", use_container_width=True):
                    df = load_data_from_file(DEFAULT_DATA_PATH)
                    if df is not None:
                        st.session_state.df = preprocess_data(df)
                        st.session_state.data_loaded = True
                        st.session_state.data_source = "Default Dataset"
                        st.sidebar.success("Default dataset loaded!")
                        st.rerun()
    
    # Model status section
    if flow in ["Prediction Only", "Both"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Status")
        
        if st.session_state.model_loaded:
            st.sidebar.success("Model: Loaded")
            st.sidebar.info(f"Features: {len(st.session_state.feature_names or VACCINE_COLS)}")
        else:
            st.sidebar.warning("Model: Not loaded")
            if st.sidebar.button("Load Model", use_container_width=True):
                load_model_artifacts()
                st.rerun()
    
    # Overall status
    st.sidebar.markdown("---")
    st.sidebar.subheader("Overall Status")
    
    status_col1, status_col2 = st.sidebar.columns(2)
    
    with status_col1:
        if st.session_state.data_loaded:
            st.markdown("**Loaded Data**")
        else:
            st.markdown("**No Data**")
    
    with status_col2:
        if st.session_state.model_loaded:
            st.markdown("**Loaded Model**")
        else:
            st.markdown("**No Model**")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 10px; font-size: 0.8rem; color: #a0a0ff;">
        <h4 style="color: #00d4ff;">About</h4>
        <p>This application predicts life expectancy based on international vaccine coverage data using machine learning.</p>
        <p style="margin-top: 10px; font-size: 0.75rem; color: #7080a0;">
        <b>Tip:</b> Place your model files in the <code>models/</code> directory before running the app.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    return flow


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application logic"""
    
    # Try to load model on startup
    if not st.session_state.model_loaded:
        load_model_artifacts()
    
    flow = create_sidebar()
    
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1>Life Expectancy Prediction Dashboard</h1>
        <p style="font-size: 1.2rem; color: #a0a0ff;">
            Explore the relationship between vaccine coverage and life expectancy
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if flow == "Visualization Only":
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Summary", "Global Overview", "Vaccine Analysis",
                "Correlations", "Country Analysis"
            ])
            
            with tab1:
                create_data_summary(df)
            with tab2:
                create_global_overview(df)
            with tab3:
                create_vaccine_coverage_analysis(df)
                st.markdown("---")
                create_scatter_analysis(df)
            with tab4:
                create_correlation_analysis(df)
            with tab5:
                create_country_analysis(df)
        else:
            st.info("Please upload your dataset or load the default dataset using the sidebar to view visualizations.")
    
    elif flow == "Prediction Only":
        create_prediction_interface()
    
    else:  # Both
        tab_pred, tab_summary, tab_global, tab_vaccine, tab_corr, tab_country = st.tabs([
            "Prediction", "Summary", "Global Overview",
            "Vaccine Analysis", "Correlations", "Country Analysis"
        ])
        
        with tab_pred:
            create_prediction_interface()
        
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            
            with tab_summary:
                create_data_summary(df)
            with tab_global:
                create_global_overview(df)
            with tab_vaccine:
                create_vaccine_coverage_analysis(df)
                st.markdown("---")
                create_scatter_analysis(df)
            with tab_corr:
                create_correlation_analysis(df)
            with tab_country:
                create_country_analysis(df)
        else:
            for tab in [tab_summary, tab_global, tab_vaccine, tab_corr, tab_country]:
                with tab:
                    st.info("Please upload your dataset or load the default dataset using the sidebar to view visualizations.")


if __name__ == "__main__":
    main()