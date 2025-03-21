import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from config import API_URL  # Import hằng số từ file cấu hình

# Cấu hình giao diện đa trang
st.set_page_config(page_title='PDF Table Detection', page_icon='📄', layout='wide')

# Sidebar cho điều hướng giữa các trang
st.sidebar.title('PDF Table Detection')
page = st.sidebar.radio('Chọn trang', ['Trang 1: Phát hiện bảng từ PDF', 'Trang 2: Tùy chỉnh'])

# Hàm gọi API detect bảng từ PDF
def detect_table_in_pdf(uploaded_file):
    try:
        response = requests.post(f"{API_URL}/api/detect_table", files={"file": uploaded_file})
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

        # Nút gọi API để detect bảng
        if st.button("Detect bảng trong PDF"):
            with st.spinner("Đang xử lý..."):
                # Gọi hàm detect bảng từ PDF
                data = detect_table_in_pdf(uploaded_file)
                if data is not None:
                    df = pd.DataFrame(data)
                    st.write("### Kết quả phân tích:")
                    st.dataframe(df)

                    # Nút download kết quả dưới dạng Excel
                    output = BytesIO()
                    df.to_excel(output, index=False)
                    output.seek(0)

                    st.download_button(
                        label="📥 Tải về kết quả dưới dạng Excel",
                        data=output,
                        file_name="result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
else:
    st.title('🚧 Trang 2: Tùy chỉnh')
    st.write("Trang này sẽ được thiết kế sau.")