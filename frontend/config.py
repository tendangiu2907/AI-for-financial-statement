# Cấu hình địa chỉ, port của backend server
SERVER_ADDRESS = "http://localhost"
SERVER_PORT = 8080
API_URL = f"{SERVER_ADDRESS}:{SERVER_PORT}"

# Cấu hình các endpoint
detect_table_endpoint = f"{API_URL}/api/v1/detect_table"
upload_extracted_path_endpoint = f"{API_URL}/api/v1/upload_extracted_path"
ask_enpoint= f"{API_URL}/api/v1/ask"
