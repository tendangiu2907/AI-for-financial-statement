import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from config import API_URL, detect_table_endpoint  # Import hằng số từ file cấu hình

# Cấu hình giao diện đa trang
st.set_page_config(page_title='PDF Table Detection', page_icon='📄', layout='wide')

# Sidebar cho điều hướng giữa các trang
st.sidebar.title('PDF Table Detection')
page = st.sidebar.radio('Chọn trang', ['Trang 1: Phát hiện bảng từ PDF', 'Trang 2: Tùy chỉnh'])

# Hàm gọi API detect bảng từ PDF
def detect_table_in_pdf(uploaded_file):
    try:
        response = requests.post(detect_table_endpoint, files={"file": uploaded_file})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Lỗi khi gọi API detect bảng: {str(e)}")
        return None

if page == 'Trang 1: Phát hiện bảng từ PDF':
    st.title('🔍 PDF Table Detection')
    st.write("Upload file PDF để phát hiện bảng và tải về kết quả dưới dạng Excel.")

    # Tải lên file PDF
    uploaded_file = st.file_uploader("Chọn file PDF", type=["pdf"])

    if uploaded_file is not None:
        st.success("Đã tải lên thành công!")

        # Tạo một biến trạng thái để quản lý nút Detect
        detect_button = st.button("Detect bảng trong PDF")

        if detect_button:
            with st.spinner("Đang xử lý..."):
                # Disable nút Detect
                st.session_state['detect_disabled'] = True

                # Gọi hàm detect bảng từ PDF
                data = detect_table_in_pdf(uploaded_file)

                # Kích hoạt lại nút Detect
                st.session_state['detect_disabled'] = False

                if data is not None:
                    st.write("### Kết quả phân tích:")

                    # Hiển thị các bảng từ kết quả trả về
                    for table_name, records in data.get("tables", {}).items():
                        st.write(f"#### Bảng: {table_name}")
                        df = pd.DataFrame(records)
                        st.dataframe(df)

                    # Nút download kết quả từ download_url
                    extracted_file_path = data.get("extracted_file_path")
                    if extracted_file_path:
                        download_url = f"{API_URL}{extracted_file_path}" 
                        st.markdown(f"[📥 Tải về kết quả tại đây]({download_url})", unsafe_allow_html=True)
                    else:
                        st.warning("Không tìm thấy đường dẫn tải về.")
else:
    st.title('🚧 Trang 2: Tùy chỉnh')
    st.write("Trang này sẽ được thiết kế sau.")