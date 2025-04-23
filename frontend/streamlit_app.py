import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from streamlit_option_menu import option_menu
from config import API_URL, detect_table_endpoint

# Enhanced Page Configuration
st.set_page_config(
    page_title='FININtel - Financial Statement Extractor', 
    page_icon='📊', 
    layout='wide'
)

# Modern and Clean CSS Styling
st.markdown("""
<style>
    /* Import Google Font Roboto */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    /* Root Variables for Easy Color Management */
    :root {
        --primary-color: #2c3e50;  /* Xanh đậm */
        --secondary-color: #3498db; /* Xanh sáng */
        --background-color: #ffffff; /* Trắng */
        --text-color: #2c3e50; /* Xanh đậm */
        --accent-color: #2980b9; /* Xanh đậm hơn */
    }

    /* Global Styling */
    .stApp {
        background-color: var(--background-color);
        font-family: 'Roboto', sans-serif;
        color: var(--text-color);
    }

    body {
        font-family: 'Roboto', sans-serif;
        color: var(--text-color);
    }

    /* Main Title Styling */
    .main-title {
        color: var(--primary-color);
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 30px;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Sidebar Styling */
    .css-1aumxhk {
        background-color: white;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }

    /* Buttons Styling */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
        color: white !important;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        font-family: 'Roboto', sans-serif;
        border: none;
        padding: 10px 20px;
    }

    .stButton > button:hover {
        background: linear-gradient(45deg, var(--primary-color), var(--accent-color)) !important;
        transform: scale(1.05);
    }

    /* File Uploader Styling */
    .css-qri22k {
        border-radius: 10px;
        border: 2px dashed var(--secondary-color);
    }

    /* Filename Styling After Upload */
    .stFileUploader div[data-testid="stFileUploaderFileName"] {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }

    /* Upload Title */
    .upload-title {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 5px;
        text-align: left;
    }

    /* Success Message Styling */
    .stSuccess {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }

    /* Uploaded Filename Styling */
    .uploaded-filename {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Subtitles */
    .sub-title {
        color: var(--primary-color);
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Download Link */
    .download-link {
        display: inline-block;
        background-color: var(--secondary-color);
        color: white !important;
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none;
        transition: background-color 0.3s ease;
        margin-top: 20px;
    }

    .download-link:hover {
        background-color: var(--accent-color);
    }
</style>

""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Menu", 
        ["Extract Financial Statements", "Dashboard", "Settings"], 
        icons=['house', 'file-earmark-spreadsheet', 'bar-chart-line', 'gear'], 
        menu_icon="cast", 
        default_index=0
    )

# Table Detection Function
def detect_table_in_pdf(uploaded_file):
    try:
        response = requests.post(detect_table_endpoint, files={"file": uploaded_file})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error calling table detection API: {str(e)}")
        return None

# Main Application Logic
if selected == 'Extract Financial Statements':
    st.markdown('<div class="main-title">FINANCIAL STATEMENTS EXTRACTOR</div>', unsafe_allow_html=True)
    
    # PDF Upload Section
    st.markdown('<div class="upload-title">Upload Financial Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"], 
        help="Upload a PDF financial statement for table extraction",
        key="pdf_uploader",
        label_visibility="collapsed"  # This hides the default label
    )
    
    if uploaded_file is not None:
        # Custom success message with dark text
        st.markdown(f'<div class="stSuccess uploaded-filename">File uploaded successfully </div>', unsafe_allow_html=True)
        
        # Detect Tables Button
        detect_button = st.button("Extract Tables", key="detect_button")
        
        if detect_button:
            with st.spinner("Analyzing document..."):
                # Call table detection API
                data = detect_table_in_pdf(uploaded_file)
                
                if data is not None:
                    st.markdown('<div class="sub-title">Extracted Tables</div>', unsafe_allow_html=True)
                    
                    # Display Extracted Tables
                    for table_name, records in data.get("tables", {}).items():
                        st.write(f"#### {table_name}")
                        df = pd.DataFrame(records)
                        st.dataframe(df)
                    
                    # Download Link
                    extracted_file_path = data.get("extracted_file_path")
                    if extracted_file_path:
                        download_url = f"{API_URL}{extracted_file_path}"
                        st.markdown(f'<a href="{download_url}" class="download-link" download>Download Extracted Data</a>', unsafe_allow_html=True)
                    else:
                        st.warning("No download link available.")

elif selected == 'Dashboard':
    st.markdown('<div class="main-title">Financial Dashboard</div>', unsafe_allow_html=True)
    st.write("Coming soon...")

elif selected == 'Settings':
    st.markdown('<div class="main-title">Application Settings</div>', unsafe_allow_html=True)
    st.write("Configuration options coming soon...")
