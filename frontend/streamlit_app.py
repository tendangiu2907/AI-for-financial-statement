import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from streamlit_option_menu import option_menu
from config import API_URL, detect_table_endpoint
from config import API_URL, ask_enpoint, upload_extracted_path_endpoint

# Enhanced Page Configuration
st.set_page_config(
    page_title='FININtel - AI For Financial Statement', 
    page_icon='📊', 
    layout='wide'
)


# Initialize session state variables if they don't exist
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Extract Financial Statements"
if "extracted_data" not in st.session_state:
    st.session_state["extracted_data"] = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "table_data" not in st.session_state:
    st.session_state.table_data = {}
if "download_url" not in st.session_state:
    st.session_state.download_url = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "extraction_completed" not in st.session_state:
    st.session_state.extraction_completed = False
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

.box {
    padding: 20px;
    border-radius: 10px;
    background-color: #f0f2f6;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
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
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 20px;
    background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Section Title Styling */
.section-title {
    color: var(--primary-color);
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 15px;
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
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 15px;
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

/* Data Preview Container */
.data-preview-container {
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 20px;
}

/* Table Header */
.table-header {
    background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 20px;
    font-weight: 600;
    margin: 15px 0 10px 0;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 4px 4px 0px 0px;
    padding: 10px 16px;
    background-color: #f1f1f1;
    border-bottom: none;
}

.stTabs [aria-selected="true"] {
    background-color: #3498db !important;
    color: white !important;
}

/* Ensure buttons in column layout are full width */
.stButton button {
    width: 100%;
}

.stDownloadButton > button {
    background-color: #0E1117;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.5rem 1.25rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
        
.stDownloadButton > button:hover {
    background-color: #1F2228;
    color: white;
}

/* Chat message styling */
[data-testid="stChatMessageContent"] {
    max-width: 80%;
}

/* User messages (right side) */
.stChatMessage [data-testid="chatAvatarIcon-user"] {
    order: 2;
}

.stChatMessage [data-testid="stChatMessageContent-user"] {
    background-color: var(--secondary-color);
    color: white;
    border-radius: 15px 0px 15px 15px;
}

/* Assistant messages (left side) */
.stChatMessage [data-testid="chatAvatarIcon-assistant"] {
    order: 0;
}

.stChatMessage [data-testid="stChatMessageContent-assistant"] {
    background-color: #f0f2f6;
    border-radius: 0px 15px 15px 15px;
}

/* Message container alignment */
.stChatMessage {
    display: flex;
    flex-direction: row;
    justify-content: flex-start;
    width: 100%;
}

/* User message container */
.user-message-container {
    display: flex;
    justify-content: flex-end;
    width: 100%;
    margin-bottom: 10px;
}

/* Assistant message container */
.assistant-message-container {
    display: flex;
    justify-content: flex-start;
    width: 100%;
    margin-bottom: 10px;
}

/* Column container styling */
.column-container {
    border-radius: 10px;
    background-color: #f8f9fa;
    padding: 15px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    height: calc(100vh - 120px);
    overflow-y: auto;
    border: 1px solid #e0e0e0;
}

/* Navigation menu styling */
.nav-menu {
    height: 100%;
    padding: 10px;
}

/* Chat container styling */
.chat-container {
    display: flex;
    flex-direction: column-reverse;  /* Đảo ngược thứ tự hiển thị */
    height: 70vh;
    overflow-y: auto;
    padding: 15px;
    background-color: white;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}

/* Chat messages container */
.chat-messages-container {
    display: flex;
    flex-direction: column-reverse;  /* Đảo ngược thứ tự hiển thị */
    height: 500px;
    overflow-y: auto;
    padding: 15px;
    background-color: white;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}

/* Tùy chỉnh thanh cuộn */
.chat-messages-container::-webkit-scrollbar,
.chat-container::-webkit-scrollbar {
    width: 6px;
}

.chat-messages-container::-webkit-scrollbar-track,
.chat-container::-webkit-scrollbar-track {
    background: #f1f1f1;
}

.chat-messages-container::-webkit-scrollbar-thumb,
.chat-container::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

.chat-messages-container::-webkit-scrollbar-thumb:hover,
.chat-container::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
}

.chat-messages {
    flex-grow: 1;
    display: flex;
    flex-direction: column-reverse;  /* Đảo ngược thứ tự hiển thị */
    overflow-y: auto;
    background-color: #fff;
    border-radius: 8px;
    height: calc(100% - 60px);
}

.chat-input {
    margin-top: 15px;
}

.user-message {
    background-color: var(--secondary-color);
    color: white;
    border-radius: 15px 0px 15px 15px;
    padding: 10px 15px;
    margin-bottom: 10px;
    display: inline-block;
    max-width: 75%;
    align-self: flex-end;  /* Đẩy tin nhắn người dùng sang phải */
    clear: both;
}

.assistant-message {
    background-color: #f0f2f6;
    border-radius: 0px 15px 15px 15px;
    padding: 10px 15px;
    margin-bottom: 10px;
    display: inline-block;
    max-width: 75%;
    align-self: flex-start;  /* Đẩy tin nhắn trợ lý sang trái */
    clear: both;
}

.input-area {
    position: relative;
    width: 100%;
    background-color: white;
    padding: 10px 0;
    border-top: 1px solid #e0e0e0;
}

/* Kiểu tin nhắn mới */
.message-row-user {
    display: flex;
    justify-content: flex-end;
    width: 100%;
    margin-bottom: 10px;
}

.message-row-assistant {
    display: flex;
    justify-content: flex-start;
    width: 100%;
    margin-bottom: 10px;
}

.message-bubble-user {
    background-color: var(--secondary-color);
    color: white;
    padding: 10px 15px;
    border-radius: 15px 0px 15px 15px;
    max-width: 80%;
}

.message-bubble-assistant {
    background-color: #f0f2f6;
    padding: 10px 15px;
    border-radius: 0px 15px 15px 15px;
    max-width: 80%;
}
</style>
""", unsafe_allow_html=True)

# Table Detection Function
def detect_table_in_pdf(uploaded_file):
    try:
        response = requests.post(detect_table_endpoint, files={"file": uploaded_file})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error calling table detection API: {str(e)}")
        return None
    
# Chat Function
def send_message_to_chatbot(message):
    try:
        # Gửi dữ liệu theo dạng form (form-urlencoded)
        data = {
            "query": message
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        response = requests.post(ask_enpoint, data=data, headers=headers)
        # response.raise_for_status()

        return response.json()

    except requests.RequestException as e:
        st.error(f"Error calling chat API: {str(e)}")
        return None


def upload_extracted_path(extracted_file_path: str) -> bool:
    """
    Gửi extracted_file_path lên API để tạo chatbot instance.
    """
    try:
        clean_path = extracted_file_path.lstrip("/")
        response = requests.post(
            upload_extracted_path_endpoint,
            data={"extracted_file_path": clean_path },  # Dùng data thay vì json
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Lỗi khi gửi extracted_file_path: {e}")
        return False

# Sidebar Navigation (use st.sidebar, not col1.st.sidebar)
with st.sidebar:
    selected = option_menu(
        "Menu", 
        ["Log in", "Sign in", "Settings"], 
        icons=['file-earmark-spreadsheet', 'person-lines-fill', 'gear'], 
        menu_icon="cast", 
        default_index = ["Log in", "Sign in", "Settings"].index("Log in")
    )
#Create  2columns
col1, col2 = st.columns(2)  # Mỗi cột chiếm 50%

#Columns1
with col1:
    st.markdown('<div class="main-title">EXTRACT FINANCIAL STATEMENTS </div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"], 
        help="Upload a PDF financial statement for table extraction",
        key="pdf_uploader",
        label_visibility="collapsed"  # This hides the default label
    )
    # Store the uploaded file in session state if it's new
    if uploaded_file is not None:
        # Save the uploaded file to session state for persistence across page changes
        if st.session_state.uploaded_file != uploaded_file:
            # Get file content before assignment to preserve it
            file_content = uploaded_file.getvalue()
            st.session_state.uploaded_file = uploaded_file
            # Reset extraction status when a new file is uploaded
            st.session_state.extraction_completed = False
            st.session_state.table_data = {}
            st.session_state.extracted_data = None
            
            # Ensure the file pointer is reset for future use
            st.session_state.uploaded_file.seek(0)
            
        # Custom success message with dark text
        st.markdown(f'<div class="stSuccess uploaded-filename">File uploaded successfully </div>', unsafe_allow_html=True)
        
        # Only show extract button if extraction hasn't been completed or new file uploaded
        if not st.session_state.extraction_completed:
            # Detect Tables Button
            detect_button = st.button("Extract Tables", key="detect_button")
            
            if detect_button:
                with st.spinner("Analyzing document..."):
                    # Make sure we're using the file from session state
                    file_to_use = st.session_state.uploaded_file
                    # Make sure file pointer is at the beginning
                    if hasattr(file_to_use, 'seek'):
                        file_to_use.seek(0)
                    
                    # Call table detection API
                    data = detect_table_in_pdf(file_to_use)
                    
                    if data is not None:
                        # Store table data in session state so it persists across page changes
                        st.session_state.table_data = data.get("tables", {})
                        st.session_state.extraction_completed = True
                        
                        # Store all tables in session state for chatbot
                        st.session_state["extracted_data"] = {
                            table_name: pd.DataFrame(records)
                            for table_name, records in st.session_state.table_data.items()
                        }
                        
                        # Save download URL
                        extracted_file_path = data.get("extracted_file_path")
                        print("Đường dẫn excel đây:",extracted_file_path)
                        if extracted_file_path:
                            st.session_state.download_url = f"{API_URL}{extracted_file_path}"
                            
                            # ✅ Gọi hàm upload_extracted_path
                            if  upload_extracted_path(f"{extracted_file_path}"):
                                st.success("File is read by chatbot")
                            else:
                                st.error("Reading file success!!")
        
        # Display results if extraction was completed (nằm ngoài if uploaded_file)
        if st.session_state.extraction_completed:
            st.markdown('<div class="sub-title">Extracted Tables</div>', unsafe_allow_html=True)
            
            # Tạo danh sách các tab dựa trên tên của các bảng
            table_names = list(st.session_state.table_data.keys())
            
            # Tạo tab container
            tabs = st.tabs(table_names)
            
            # Hiển thị DataFrame trong mỗi tab
            for i, (table_name, records) in enumerate(st.session_state.table_data.items()):
                with tabs[i]:
                    df = pd.DataFrame(records)
                    st.dataframe(df)

        # Prepare Excel file with all tables (outside file upload condition)
        if st.session_state.extraction_completed and st.session_state.table_data:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                for table_name, records in st.session_state.table_data.items():
                    df = pd.DataFrame(records)
                    # Đảm bảo tên bảng hợp lệ cho sheet Excel (tối đa 31 ký tự, không có ký tự đặc biệt)
                    safe_name = table_name[:31].replace('/', '').replace('\\', '').replace('?', '').replace('*', '')
                    if not safe_name:  # Đảm bảo có ít nhất một ký tự
                        safe_name = f"Table_{list(st.session_state.table_data.keys()).index(table_name)}"
                    df.to_excel(writer, sheet_name=safe_name, index=False)
    
            excel_bytes = excel_buffer.getvalue()
            st.download_button(
                "Tải xuống",
                data=excel_bytes,
                file_name="extracted_financial_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


with col2:
    st.markdown('<div class="main-title">FINANCIAL ASSISTANTS</div>', unsafe_allow_html=True)
    
    # Tạo container cho khu vực chat với chiều cao cố định
    chat_container = st.container()
    
    # Tạo container riêng cho phần nhập liệu
    input_container = st.container()
    
    # Xử lý đầu vào từ người dùng trước (ở dưới cùng của màn hình)
    with input_container:
        user_input = st.chat_input("Ask a question about your financial statements...")
        
        if user_input:
            # Thêm tin nhắn người dùng vào lịch sử
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Xử lý phản hồi từ chatbot
            with st.spinner("..."):
                result = send_message_to_chatbot(user_input)
                
                if result:
                    response = result.get("response", "No response received from the chatbot.")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    error_message = "⚠️ Error connecting to the chatbot. Please try again later."
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                
    
    # Hiển thị lịch sử chat trong container chính
    with chat_container:
        chat_html = """
        <div style="height: 500px; overflow-y: auto; display: flex; flex-direction: column-reverse;">
        """

        for message in reversed(st.session_state.messages):
            if message["role"] == "user":
                chat_html += (
                    f'<div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">'
                    f'<div style="background-color: #3498db; color: white; padding: 10px 15px; '
                    f'border-radius: 15px 0px 15px 15px; max-width: 80%;">'
                    f'{message["content"]}'
                    f'</div></div>'
                )
            else:
                chat_html += (
                    f'<div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">'
                    f'<div style="background-color: #f0f2f6; padding: 10px 15px; '
                    f'border-radius: 0px 15px 15px 15px; max-width: 80%;">'
                    f'{message["content"]}'
                    f'</div></div>'
                )

        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
