import time
import os
import json
import cv2
# import re
# import torch
import numpy as np
import pandas as pd
import tensorflow as tf
# import supervision as sv
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.patches import Patch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR, draw_ocr
from pdf2image import convert_from_path
from google import genai
from google.genai import types
from unidecode import unidecode
from fuzzywuzzy import fuzz
from datetime import datetime
# import base64
# import tempfile

from core.config import MODEL_SIGNATURE_PATH, MODEL_TABLE_TITLE_PATH, DEVICE, POPPLER_PATH, financial_tables, model, EXTRACTED_FOLDER, financial_tables_general
from utils import retry_api_call, dataframe_to_json, json_to_dataframe_table, json_to_dataframe_title
from secret import api_keys


class TableDetectService:
    def __init__(self):
        print("TableDetectService: Khởi tạo là load model...")
        self.ocr = PaddleOCR(lang="en")
        self.table_title_detection_model = YOLO(MODEL_TABLE_TITLE_PATH).to(DEVICE)
        self.signature_detection_model = YOLO(MODEL_SIGNATURE_PATH).to(DEVICE)
        self.detection_class_names = ["table", "table rotated"]
        self.structure_class_map = {
            k: v
            for v, k in enumerate(
                [
                    "table",
                    "table column",
                    "table row",
                    "table column header",
                    "table projected row header",
                    "table spanning cell",
                    "no object",
                ]
            )
        }
        self.structure_class_thresholds = {
            "table": 0.5,
            "table column": 0.5,
            "table row": 0.5,
            "table column header": 0.5,
            "table projected row header": 0.5,
            "table spanning cell": 0.5,
            "no object": 10,  # Giá trị cao để loại bỏ "no object"
        }

    def detect_table(self, pdf_path, file_name_origin):
        """
        Flow xử lý file pdf như sau:
        - Chuyển file pdf thành danh sách các hình ảnh
        - Lặp qua từng hình và xử lý như sau:
            + Nếu hình đó có chứa table:
                > Sử dụng model best_model_YOlO để detect ra bẳng
        """
        recognized_titles_set = set()
        dfs_dict = {}


        images = self.pdf_to_images(pdf_path)  # Chuyển pdf thành hình ảnh

        index_start = 0  # Bắt đầu từ ảnh đầu tiên
        while index_start < len(images):
            index_chuky = None  # Reset mỗi lần lặp
            for i in range(index_start, len(images)):
                selected_images = []
                image = images[i]
                print(f"======== BẮT ĐẦU XỬ LÝ ẢNH {i+1} ========")


                # Nhận diện bảng -> table-title
                print(f"==== Kiểm tra bảng trong ảnh ====")
                nhandien_table = self.table_detection(image)

                if not nhandien_table:
                    print(f"==== Ảnh không có bảng, chuyển sang ảnh tiếp theo ====")
                    print(f"======== KÉT THÚC XỬ LÝ ẢNH {i+1} NO_TABLE ========\n\n\n\n")
                    continue  # Nếu không có bảng, bỏ qua ảnh này

                has_rotated_table = any(
                    self.detection_class_names[det[5]] == "table rotated"
                    for det in nhandien_table
                )
                
                # Chỉ xoay ảnh nếu có bảng xoay
                image_to_process = (
                    self.table_rotation(image, nhandien_table) if has_rotated_table else image
                )

                print(f"==== Nhận diện title của bảng ====")
                df_title, text_title = self.detect_and_extract_title(image_to_process)
                for api_key in api_keys:
                    json_title = retry_api_call(
                        self.generate_title,
                        model,
                        api_keys[api_key]["title"],
                        dataframe_to_json(df_title),
                        text_title)
                    if json_title:
                        break
                print("==== Hoàn tất thử API cho nhận diện title ====")
                print("==== Kết quả title ====")
                print(f"{json_title}")

                data_title = json_to_dataframe_title(json_title)  # Kết quả title của bảng
                recognized_title = self.recognize_financial_table(
                    data_title, financial_tables_general, threshold=80
                )  # Nhận diện xem title của bảng là gì có phù hợp với 3 tên bảng dự án đề ra không

                # Nếu nhận diện được title, thêm vào danh sách nhận diện
                if not (recognized_title):
                    print(f"==== Không tìm thấy title trong ảnh ====")
                    print(f"======== KÉT THÚC XỬ LÝ ẢNH {i+1} NO_TITLE========\n\n\n\n")
                    # Để sleep để giúp model nghỉ, bị limit 1 phút không quá 2 lần
                    time.sleep(45)
                    continue

                print(f"==== Nhận diện được title của ảnh là : {recognized_title} ====")

                # Tìm ảnh chữ ký tiếp theo sau ảnh title
                print(f"==== Nhận diện chữ kí từ ảnh tiếp theo ====")
                for j in range(images.index(image), len(images)):
                    nhandien_chuky = images[j]
                    results_chuky = self.detect_signature(nhandien_chuky)
                    if results_chuky[0]:
                        index_chuky = j  # Lưu vị trí ảnh chữ ký
                        print(f"==== Ảnh chữ ký được phát hiện ở ảnh thứ {index_chuky +1 } ====")
                        break

                # Lấy danh sách ảnh từ title đến chữ ký
                if index_chuky:
                    selected_images.extend(images[images.index(image) : index_chuky + 1])

                print(f"==== Cho model giải lao trước khi nhận diện thông tin bảng ====")
                time.sleep(45)

                # Vòng lặp qua ảnh từ title đến chữ ký để trích xuất bảng
                if selected_images:
                    print(f"==== Nhận diện thông tin của bảng {recognized_title} ====")
                    pre_name_column = None
                    for img in selected_images:
                        processed_image = self.Process_Image(img)
                         # 2️⃣ Chuyển đổi ảnh sang CMYK và lấy kênh K
                        _, _, _, black_channel = self.rgb_to_cmyk(processed_image)
                        # 3️⃣ Điều chỉnh độ sáng & độ tương phản
                        processed_image = self.adjust_contrast(black_channel, alpha=2.0, beta=-50)
                        if processed_image is not None:
                            df_table, text_table = self.process_pdf_image(processed_image)
                            print(f"Check json data: ", dataframe_to_json(df_table))
                            print()
                            if not df_table.empty:
                                if (len(df_table) < 101) and (len(df_table.columns) < 10):
                                    token = 9000
                                elif (len(df_table) < 201) and (len(df_table.columns) < 10):
                                    token = 18000
                                else:
                                    token = 30000
                                if selected_images.index(img) ==0:
                                    response_schema=self.generate_json_schema(dataframe_to_json(df_table))
                                for api_key in api_keys:
                                    json_table = retry_api_call(
                                        self.generate_table_1,
                                        model,
                                        api_keys[api_key]["table"],
                                        dataframe_to_json(df_table),
                                        text_table,
                                        token,
                                        pre_name_column, response_schema)
                                    if json_table:
                                        break
                                print("==== Hoàn tất thử API cho nhận diện thông tin của bảng ====")
                                print(f"==== Kết quả thông tin của bảng {recognized_title} ====")
                                print(json_table)    

                                data_table = json_to_dataframe_table(json_table)

                                if selected_images.index(img) ==0:
                                    found = False  # Flag để thoát cả hai vòng lặp khi tìm thấy kết quả
                                    recognized_title = "Bảng cân đối kế toán"
                                    for column in data_table.columns:
                                        for value in data_table[column].dropna():
                                            value = self.normalize_text(value)

                                            if "luu chuyen" in value:
                                                recognized_title = "Báo cáo lưu chuyển tiền tệ"
                                                found = True
                                                break  # Thoát khỏi vòng lặp giá trị trong cột

                                            elif "doanh thu ban hang" in value or "ban hang" in value:
                                                recognized_title = "Báo cáo KQHĐKD"
                                                found = True
                                                break  # Thoát khỏi vòng lặp giá trị trong cột
                                        if found:
                                            break  # Thoát khỏi vòng lặp cột

                                recognized_titles_set.add(recognized_title)
                                # display(data_table)
                                if selected_images.index(img) == 0:
                                    pre_name_column = data_table.columns.tolist()
                                else:
                                    if len(data_table.columns) == len(pre_name_column):
                                        data_table.columns = pre_name_column
                                    else:
                                        data_table = data_table.reindex(
                                            columns=pre_name_column, fill_value=None
                                        )

                                if not data_table.empty:
                                    if recognized_title not in dfs_dict:
                                        dfs_dict[recognized_title] = data_table
                                    else:
                                        dfs_dict[recognized_title] = pd.concat(
                                            [dfs_dict[recognized_title], data_table],
                                            ignore_index=True,
                                        )
                        time.sleep(45)
                            
                    print(f"==== Hoàn tất nhận diện thông tin bảng {recognized_title} ====")
                    print(f"======== KÉT THÚC XỬ LÝ ẢNH {i+1} SUCCESS========\n\n\n\n")
                    break # beak để cập nhật lại ví trí bắt đầu là vị trí kế tiếp của ảnh có chữ kí
                
            # Cập nhật vị trí bắt đầu cho vòng lặp tiếp theo
            if index_chuky:
                index_start = index_chuky + 1
            else:
                index_start = i + 1
                # Kiểm tra nếu đã nhận diện đủ bảng tài chính thì dừng
            if recognized_titles_set == set(financial_tables):
                print("======== ĐÃ NHẬN DIỆN ĐỦ TẤT CẢ CÁC BẢNG TÀI CHÍNH. DỪNG LẠI !! ========\n\n\n\n")
                break

        # Lưu kết quả vào file Excel
        print(f"======== BẤT ĐẦU LƯU DỮ LIỆU VÀO FILE ========")
        name, _ = file_name_origin.rsplit(".", 1) if "." in file_name_origin else (file_name_origin, "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Định dạng thời gian: YYYYMMDD_HHMMSS
        new_name = f"{name}_{timestamp}.xlsx"
        file_path = os.path.join(EXTRACTED_FOLDER, new_name)
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer: # TODO: No module named 'xlsxwriter'
            for i, (sheet_name, df) in enumerate(dfs_dict.items()):
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                print(f"==== Đã ghi xong bảng {sheet_name[:31]} vào file ====")
        print(f"======== DỮ LIỆU ĐÃ ĐƯỢC LƯU VÀO {file_path} ========")

        download_url = f"/{EXTRACTED_FOLDER}/{new_name}"
        return dfs_dict, download_url
    
    def rgb_to_cmyk(self,image):
        """ Chuyển đổi ảnh từ RGB sang không gian màu CMYK """
        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.uint8)
        b, g, r = cv2.split(image)
        # Chuyển giá trị pixel về khoảng [0,1]
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

        # Tính toán kênh K (đen)
        k = 1 - np.max([r, g, b], axis=0)

        # Tránh chia cho 0
        c = (1 - r - k) / (1 - k + 1e-10)
        m = (1 - g - k) / (1 - k + 1e-10)
        y = (1 - b - k) / (1 - k + 1e-10)

        # Đưa về khoảng giá trị 0-255
        c = (c * 255).astype(np.uint8)
        m = (m * 255).astype(np.uint8)
        y = (y * 255).astype(np.uint8)
        k = (k * 255).astype(np.uint8)

        return c, m, y, k

    def adjust_contrast(self,image, alpha=2.0, beta=-50):
        """ Điều chỉnh độ tương phản và độ sáng """
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
    def pdf_to_images(self, pdf_path):
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        return images

    # model table_title_detection_model
    def table_detection(self, image):
        imgsz = 800
        pred = self.table_title_detection_model.predict(image, imgsz=imgsz)
        pred = pred[0].boxes
        result = pred.cpu().numpy()
        result_list = [
            list(result.xywhn[i]) + [result.conf[i], int(result.cls[i])]
            for i in range(result.shape[0])
        ]
        return result_list

    def table_rotation(self, image, list_detection_table):
        for det in list_detection_table:
            x_center, y_center, w_n, h_n, conf, cls_id = det
            if self.detection_class_names[cls_id] == "table rotated":
                print("This is a rotated table")
                image = image.rotate(-90, expand=True)
            img = image.convert("L")
            thresh_img = img.point(lambda p: 255 if p > 120 else 0)
            return thresh_img

    # model table_title_detection_model
    def Process_Image(self, image):
        results = self.table_title_detection_model.predict(image, task="detect")
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            class_name = self.table_title_detection_model.names[cls_id]

            # Chuyển đổi sang PIL nếu cần
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Cắt ảnh trước
            cropped_table = image.crop((int(x1), int(y1), int(x2), int(y2)))

            # plt.imshow(cropped_table)
            # plt.title("Cropped Image")
            # plt.show()

            # Nếu là bảng bị xoay, xoay lại
            if class_name == "table rotated":
                print("This is a rotated table")
                cropped_img = cropped_table.rotate(-90, expand=True)
                return cropped_img  # Trả về ảnh đã cắt và sửa góc

            return cropped_table  # Nếu không bị xoay, trả về ảnh cắt nguyên bản

    # sử dụng model từ models = ["gemini-2.0-pro-exp-02-05", "gemini-2.0-flash-thinking-exp-01-21"]
    # chuyển API key sang file config
    def generate_title(self, model, API, path_title_json, text_title):
        result = ""
        client = genai.Client(api_key=f"{API}")

        # Mở file JSON và đọc nội dung
        file_path = path_title_json
        with open(file_path, "r", encoding="utf-8") as f:
            json_content = json.load(f)  # Load JSON thành dict

        model = model
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"""Mình đang trích xuất dữ liệu từ hình ảnh chứa bảng tài chính bằng PaddleOCR. Dữ liệu nhận diện được lưu trong {text_title}.
    Tuy nhiên, dữ liệu gặp lỗi:
    - Sai chính tả tiếng Việt trong báo cáo tài chính, kế toán và dòng tiền
    - Lỗi ngữ pháp tiếng Việt trong báo cáo tài chính, kế toán và dòng tiền
    Vì đây là một báo cáo quan trọng, rất nhiều thứ ảnh hưởng xấu đến nếu như nó sai chính tả và lỗi ngữ pháp.
    Bạn hãy trả về cho mình một DataFrame chỉ có 1 cột là "values" chứa các giá trị được ngăn cách thành từng dòng giúp người đọc dễ dàng đọc hiêu, mỗi hàng không chứa lồng ghép thành chuỗi hay danh sách gì, chỉ 1 dòng là 1 giá trị riêng biệt từ file JSON gốc.
                        Dữ liệu JSON gốc:
                        {json.dumps(json_content, indent=2, ensure_ascii=False)}
                        """
                    ),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            max_output_tokens=8192,
            response_mime_type="application/json",
        )

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result += chunk.text
        return result
    def generate_json_schema(self, json_file_path):
        """Tạo JSON Schema từ file JSON đầu vào."""

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file JSON tại '{json_file_path}'")
            return
        except json.JSONDecodeError:
            print(f"Lỗi: File '{json_file_path}' không phải là JSON hợp lệ.")
            return

        json_string = json.dumps(json_data, ensure_ascii=False)

        client = genai.Client(api_key="AIzaSyAVa_jH5PG6UnOIpTD0MQztdI4QEPIKs5Y")
        model = "gemini-2.0-flash"

        prompt = f"""
        Hãy tạo một JSON Schema từ dữ liệu JSON sau đây:

        {json_string}

        Yêu cầu:
        1. Tạo một JSON Schema hợp lệ để mô tả cấu trúc dữ liệu JSON.
        2. Sử dụng các kiểu dữ liệu JSON Schema phù hợp cho từng thuộc tính.
        3. Nếu có thể, hãy suy ra các ràng buộc (constraints) từ dữ liệu (ví dụ: required, nullable).
        4. Tạo một schema tổng quát, có thể xử lý các JSON có cấu trúc khác nhau.
        5. Trả về kết quả là một chuỗi JSON Schema hợp lệ.
        """

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text

        try:
            # Kiểm tra và in JSON Schema hợp lệ
            json.loads(response_text)
        except json.JSONDecodeError:
            print("Lỗi: Kết quả không phải là JSON hợp lệ.")
            print("Kết quả từ Gemini:")

    def generate_table(self, model, API, path_dataframe_json, text_table, token, table_columns, response_schema):
        """Tạo bảng dữ liệu từ JSON đầu vào, sử dụng JSON Schema trả về."""

        result = ""

        # Khởi tạo API Client
        client = genai.Client(api_key=API)

        # Mở file JSON và đọc nội dung
        with open(path_dataframe_json, "r", encoding="utf-8") as f:
            json_content = json.load(f)

        # Prompt cải tiến
        prompt_text = f"""
        Mình đang xử lý dữ liệu từ hình ảnh chứa bảng tài chính, trích xuất bằng PaddleOCR.
        Dữ liệu nhận diện được lưu trong {text_table}, nhưng gặp lỗi sai chính tả, ngữ pháp, và cấu trúc bảng.

        Dựa vào bố cục và nội dung JSON gốc, bạn hãy:
        - Sửa lỗi chính tả, ngữ pháp tiếng Việt.
        - Sắp xếp lại dữ liệu để đảm bảo đúng thứ tự dòng/cột theo chuẩn báo cáo tài chính.
        - Đặt tên cột đúng chuẩn. Nếu danh sách {table_columns} rỗng, đặt mặc định gồm "Mã số", "Tên chỉ tiêu", "Thuyết minh".
        - Chuẩn hóa dữ liệu số (định dạng số nguyên/thập phân, đơn vị tiền tệ).
        - Định dạng các cột số theo tên chỉ tiêu (ví dụ: khoản chi phí hiển thị trong dấu '()' hoặc '-').
        - Nhận diện khoảng thời gian tài chính và đặt tên cột số liệu theo thời gian được nhận diện (Ví dụ "Năm 2022", "Năm 2023").

        Ví dụ về JSON đầu ra mong muốn:
        {{
            "dataframe": [
                {{
                    "Mã số": "123",
                    "Tên chỉ tiêu": "Doanh thu bán hàng",
                    "Thuyết minh": "Doanh thu từ hoạt động bán hàng",
                    "Năm 2022": 1000000,
                    "Năm 2023": 1200000
                }},
                {{
                    "Mã số": "456",
                    "Tên chỉ tiêu": "Chi phí quản lý",
                    "Thuyết minh": "Chi phí quản lý doanh nghiệp",
                    "Năm 2022": -200000,
                    "Năm 2023": -250000
                }}
            ]
        }}

        Dữ liệu JSON gốc:
        {json.dumps(json_content, indent=2, ensure_ascii=False)}
        """

        # Cấu hình request đến API
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])
        ]

        generate_content_config = types.GenerateContentConfig(
            max_output_tokens=token,
            response_mime_type="application/json",
            response_schema=response_schema  # Sử dụng schema truyền vào
        )

        # Gửi yêu cầu và nhận kết quả
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result += chunk.text

        return result

    def process_image_ocr(self, image):
        """Nhận diện text trong ảnh bằng OCR."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        output = self.ocr.ocr(image)[0]
        boxes = [line[0] for line in output]
        texts = [line[1][0] for line in output]
        probabilities = [line[1][1] for line in output]
        return image, boxes, texts, probabilities

    def get_horizontal_vertical_boxes(self, image, boxes):
        """Tạo danh sách các bounding box ngang và dọc."""
        image_height, image_width = image.shape[:2]
        horiz_boxes = []
        vert_boxes = []

        for box in boxes:
            x_h, x_v = 0, int(box[0][0])
            y_h, y_v = int(box[0][1]), 0
            width_h, width_v = image_width, int(box[2][0] - box[0][0])
            height_h, height_v = int(box[2][1] - box[0][1]), image_height

            horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
            vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])

        return horiz_boxes, vert_boxes

    def apply_non_max_suppression(self, boxes, scores, image):
        """Áp dụng Non-Max Suppression (NMS) để loại bỏ các bounding box dư thừa."""
        nms_indices = tf.image.non_max_suppression(
            boxes,
            scores,
            max_output_size=1000,
            iou_threshold=0.1,
            score_threshold=float("-inf"),
        ).numpy()
        return np.sort(nms_indices)

    def intersection(self, box_1, box_2):
        """Tính toán giao giữa hai bbox."""
        return [box_2[0], box_1[1], box_2[2], box_1[3]]

    def iou(self, box_1, box_2):
        """Tính chỉ số Intersection over Union (IoU)."""
        x_1, y_1 = max(box_1[0], box_2[0]), max(box_1[1], box_2[1])
        x_2, y_2 = min(box_1[2], box_2[2]), min(box_1[3], box_2[3])
        inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))

        if inter == 0:
            return 0
        box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
        box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

        return inter / float(box_1_area + box_2_area - inter)

    def extract_table_data(self, boxes, texts, horiz_lines, vert_lines, horiz_boxes, vert_boxes):
        """Trích xuất dữ liệu bảng từ bbox đã nhận diện."""
        out_array = [["" for _ in range(len(vert_lines))] for _ in range(len(horiz_lines))]

        unordered_boxes = [vert_boxes[i][0] for i in vert_lines]
        ordered_boxes = np.argsort(unordered_boxes)

        for i in range(len(horiz_lines)):
            for j in range(len(vert_lines)):
                resultant = self.intersection(
                    horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]]
                )
                for b, box in enumerate(boxes):
                    the_box = [box[0][0], box[0][1], box[2][0], box[2][1]]
                    if self.iou(resultant, the_box) > 0.1:
                        out_array[i][j] = texts[b]

        return pd.DataFrame(np.array(out_array))

    def process_pdf_image(self, image):
        """Hàm tổng hợp để xử lý ảnh từ PDF, nhận diện bảng và trích xuất dữ liệu."""
        # OCR trích xuất text & bbox
        image, boxes, texts, probabilities = self.process_image_ocr(image)

        # Nhận diện box ngang & dọc
        horiz_boxes, vert_boxes = self.get_horizontal_vertical_boxes(image, boxes)

        # Loại bỏ các box dư thừa bằng Non-Max Suppression
        horiz_lines = self.apply_non_max_suppression(horiz_boxes, probabilities, image)
        vert_lines = self.apply_non_max_suppression(vert_boxes, probabilities, image)

        # Trích xuất dữ liệu bảng thành DataFrame
        df = self.extract_table_data(
            boxes, texts, horiz_lines, vert_lines, horiz_boxes, vert_boxes
        )

        return df, texts

    # model nhận diện chữ kí
    def detect_signature(self, image):
        return self.signature_detection_model(image, conf=0.7)

    # model nhận diện table title
    def detect_and_extract_title(self, image):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        
        # Nhận diện table trong ảnh để cắt phần title
        results = self.table_title_detection_model(image)

        # Lấy ảnh gốc
        img_last = results[0].orig_img.copy()

        # Lấy danh sách tọa độ chữ ký (x1, y1, x2, y2)
        boxes_obj = results[0].boxes
        if boxes_obj is not None and len(boxes_obj) > 0:
            coords = boxes_obj.xyxy.cpu().numpy()  # Chuyển về numpy array
            x1, y1, x2, y2 = map(int, coords[0])  # Lấy tọa độ đầu tiên (nếu có nhiều)

            # Lấy kích thước ảnh
            h, w, _ = img_last.shape

            # Cắt vùng trên và dưới của chữ ký
            top_region = img_last[0:y1, 0:w]
            bottom_region = img_last[y2:h, x1:x2]

            # Nhận diện văn bản từ hai vùng
            top_text = self.ocr.ocr(top_region)[0]
            bottom_text = self.ocr.ocr(bottom_region)[0]

            # Lọc kết quả nhận diện
            top_result = [
                line[1][0]
                for line in (top_text or [])  # Nếu None thì chuyển thành list rỗng
                if line and len(line) > 1 and line[1] and len(line[1]) > 0
            ]

            bottom_result = [
                line[1][0]
                for line in (bottom_text or [])  # Nếu None thì chuyển thành list rỗng
                if line and len(line) > 1 and line[1] and len(line[1]) > 0
            ]

            # Gộp kết quả từ cả hai vùng
            extracted_text = top_result + bottom_result
        else:
            extracted_text = []
        df_title = pd.DataFrame(extracted_text)
        return df_title, extracted_text

    def normalize_text(self, text):
        return unidecode(str(text)).lower().strip()

    def recognize_financial_table(self, df, financial_tables, threshold=80):
        """
        Nhận diện tiêu đề bảng tài chính từ một DataFrame.

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu cần kiểm tra.
            financial_tables (list): Danh sách các bảng tài chính chuẩn.
            image : Ảnh đang xét
            threshold (int): Ngưỡng độ tương đồng tối thiểu để chấp nhận.
        Returns:
            tuple: (Tên bảng tài chính nhận diện được, ảnh tương ứng)
        """
        # Chuẩn hóa danh sách bảng tài chính
        normalized_tables = [self.normalize_text(table) for table in financial_tables]

        # Duyệt qua từng cột trong DataFrame
        for column in df.columns:
            for value in df[column].dropna():  # Bỏ qua giá trị NaN
                norm_value = self.normalize_text(value)

                # Kiểm tra khớp chính xác trước
                if norm_value in normalized_tables:
                    print(f"✅ Khớp chính xác: {value} (cột: {column})")
                    recognized_title = financial_tables[normalized_tables.index(norm_value)]
                    return recognized_title

                # Nếu không khớp chính xác, kiểm tra độ tương đồng
                for norm_table in normalized_tables:
                    similarity = fuzz.partial_ratio(norm_value, norm_table)
                    if similarity >= threshold:
                        print(
                            f"🔹 Khớp tương đồng ({similarity}%): {value} ~ {norm_table} (cột: {column})"
                        )
                        recognized_title = financial_tables[
                            normalized_tables.index(norm_table)
                        ]
                        return recognized_title

        print("❌ Không tìm thấy bảng tài chính nào phù hợp.")
        return None

    def get_model_params(self, model):
        if model == "gemini-2.0-flash":
            return 1, 0.95, 64
        return None

    def generate_table_1(self, model, API, path_dataframe_json, text_table, token, table_columns):
        result = ""
        client = genai.Client(api_key=f"{API}")

        # Mở file JSON và đọc nội dung
        with open(path_dataframe_json, "r", encoding="utf-8") as f:
            json_content = json.load(f)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"""Mình đang trích xuất dữ liệu từ hình ảnh chứa bảng tài chính bằng PaddleOCR. Dữ liệu nhận diện được lưu trong {text_table}.
    Tuy nhiên, dữ liệu gặp lỗi:
    - Sai chính tả tiếng Việt
    - Lỗi ngữ pháp tiếng Việt
    - Sắp xếp sai dòng/cột, ảnh hưởng đến tính chính xác của báo cáo tài chính.

    Bạn hãy giúp mình chuẩn hóa lại bảng dữ liệu dựa vào bối cảnh và ký tự nhận diện được trong {text_table} và kiến thức chuyên ngành tài chính, kế toán, đảm bảo đúng thuật ngữ, chính tả và cấu trúc bảng hợp lý (gồm dòng, cột, tiêu đề cột, dữ liệu trong bảng). Yêu cầu kết quả trả về chuẩn định dạng DataFrame không gặp bất kỳ lỗi nào chuẩn theo đúng định dạng bảng báo cáo tài chính cho người dùng dễ dàng đọc hiểu,
    đảm bảo đúng thông tin được truyền vào từ biến {text_table} và Dữ liệu JSON gốc không sai kết quả.
    Đây là báo cáo kết quả hoạt động kinh doanh của công ty ABC.
    Bạn hãy kiểm tra nếu danh sách tên cột {table_columns} rỗng thì hãy nhận diện để đặt tên cột mặc định bắt buộc phải có chứa 3 cột: "Mã số", "Tên chỉ tiêu", "Thuyết minh" và chuẩn hóa các cột sau: "Mã số", "Tên chỉ tiêu", "Thuyết minh".
    Nếu danh sách tên cột {table_columns} không rỗng thì hãy đặt tên cột giống như từng giá trị trong {table_columns} và chuẩn hóa chúng đúng với kiến thức quan trọng cần thiết trong báo cáo tài chính.
    Tự động nhận diện và chuẩn hóa các cột số liệu, đảm bảo chúng được hiển thị đúng định dạng (ví dụ: số nguyên, số thập phân, đơn vị tiền tệ).
    Nếu có thể, hãy xác định năm tài chính được đề cập trong báo cáo và sử dụng thông tin này để đặt tên cho các cột số liệu (ví dụ: "Năm 2022", "Năm 2023").
    Sử dụng tên cột có dấu cách và viết hoa chữ cái đầu tiên của mỗi từ.

    Dữ liệu JSON gốc:
    {json.dumps(json_content, indent=2, ensure_ascii=False)}
    """
                    ),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            max_output_tokens=token,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "Bảng cân đối kế toán": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Mã Số": {"type": "integer"},
                                "Tên Chỉ Tiêu": {"type": "string"},
                                "Thuyết Minh": {"type": "string"},
                                "Số cuối năm": {"type": "string"},
                                "Số đầu năm": {"type": "string"}
                            },
                            "required": ["Mã Số", "Tên Chỉ Tiêu", "Thuyết Minh", "Số cuối năm", "Số đầu năm"]
                        }
                    },
                    "Báo cáo kết quả hoạt động kinh doanh": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Mã Số": {"type": "integer"},
                                "Tên Chỉ Tiêu": {"type": "string"},
                                "Thuyết Minh": {"type": "number"},
                                "Số Năm Nay": {"type": "number"},
                                "Số Năm Trước": {"type": "number"}
                            },
                            "required": ["Mã Số", "Tên Chỉ Tiêu", "Thuyết Minh", "Số Năm Nay", "Số Năm Trước"]
                        }
                    },
                    "Báo cáo lưu chuyển tiền tệ": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Mã Số": {"type": "integer"},
                                "Tên Chỉ Tiêu": {"type": "string"},
                                "Thuyết Minh": {"type": "number"},
                                "Số Năm Nay": {"type": "number"},
                                "Số Năm Trước": {"type": "number"}
                            },
                            "required": ["Mã Số", "Tên Chỉ Tiêu", "Thuyết Minh", "Số Năm Nay", "Số Năm Trước"]
                        }
                    }
                },
                "required": ["Bảng cân đối kế toán", "Báo cáo kết quả hoạt động kinh doanh", "Báo cáo lưu chuyển tiền tệ"]
            }
        )

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result += chunk.text
        return result
