import time
import tempfile
import pandas as pd
import json
import shutil
from fastapi import UploadFile
import os
from core.config import ALLOWED_EXTENSIONS
import re

def save_temp_pdf(file: UploadFile, upload_dir: str) -> str:
    """Lưu file PDF vào thư mục chỉ định."""
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path

def retry_api_call(func, *args, **kwargs):
    function_name = func.__name__  # Lấy tên hàm

    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = str(e)
            if "429" in error_message:
                print(f"[{function_name}] Lỗi 429: Thử lại API khác sau 10 giây...")
                time.sleep(10)
                return None  # Chuyển API khác
            elif any(code in error_message for code in ["500", "503", "504"]):
                print(f"[{function_name}] Lỗi server, thử lại sau 10 giây...")
                time.sleep(10)
                continue  # Thử lại API này
            else:
                print(f"[{function_name}] Lỗi khác: {e}")
                print("Đã hết API, rất xin lỗi quý khách")
                return None

def save_to_excel(final_dict):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        with pd.ExcelWriter(tmp.name, engine="xlsxwriter") as writer: # TODO: No module named 'xlsxwriter'
            for sheet_name, df in final_dict.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        return tmp.name

def dataframe_to_json(df, filename="output.json"):
    df.to_json(filename, orient="columns", indent=4)
    return filename

def json_to_dataframe(json_file):
    data = json.loads(json_file)
    return pd.DataFrame(data)

def clean_dataframe(df):
    df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
    df = df.astype(object).where(pd.notna(df), None)
    return df

def find_duplicate_column_pairs(df):
    column_duplicate_pairs = set()
    for index, row in df.iterrows():
        for col in df.columns:
            value = row[col]
            try:
                cleaned_value = re.sub(r"\D", "", value)
                float(cleaned_value)
                continue
            except (ValueError, TypeError):
                pass
            if pd.isna(value) or value == "":
                continue
            duplicate_cols = [c for c in df.columns if c != col and row[c] == value]
            for dup_col in duplicate_cols:
                pair = tuple(sorted((col, dup_col)))
                column_duplicate_pairs.add(pair)
    return column_duplicate_pairs

def count_duplicates(df, column_duplicate_pairs):
    duplicate_counts = {}
    for col_root, col_dup in column_duplicate_pairs:
        duplicate_rows = df[df[col_root] == df[col_dup]]
        duplicate_count = len(duplicate_rows)
        duplicate_counts[(col_root, col_dup)] = duplicate_count
    return duplicate_counts

def build_merge_map(duplicate_counts, threshold=30):
    column_merge_map = {}
    for (col_root, col_dup), count in duplicate_counts.items():
        if count > threshold:
            if col_root not in column_merge_map:
                column_merge_map[col_root] = set()
            column_merge_map[col_root].add(col_dup)
    return column_merge_map

def merge_columns(df, column_merge_map):
    df_merged = df.copy()
    for col_root, col_dups in column_merge_map.items():

        def merge_values(row):
            values = set()
            if pd.notna(row[col_root]) and row[col_root] != "":
                values.add(row[col_root])
            for col in col_dups:
                if pd.notna(row[col]) and row[col] != "":
                    values.add(row[col])
            return list(values)[0] if len(values) == 1 else ", ".join(map(str, values))

        df_merged[col_root] = df.apply(merge_values, axis=1)
    df_merged = df_merged.drop(
        columns=[col for cols in column_merge_map.values() for col in cols]
    )
    return df_merged

def process_dataframe(df):
    df = clean_dataframe(df)
    column_duplicate_pairs = find_duplicate_column_pairs(df)
    duplicate_counts = count_duplicates(df, column_duplicate_pairs)
    column_merge_map = build_merge_map(duplicate_counts)
    df_merged = merge_columns(df, column_merge_map)
    return df_merged

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS