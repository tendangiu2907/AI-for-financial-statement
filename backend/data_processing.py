#data_processor.py
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def convert_excel_to_json(data_path):
    
    data = pd.read_excel(data_path, sheet_name=None)
    json_output = {sheet: df.fillna("").to_dict(orient="records") for sheet, df in data.items()}
    return json_output

def load_financial_data(data_path): 
    
    json_output = convert_excel_to_json(data_path)
    
    documents = []
    for table_name, rows in json_output.items():
        for row in rows:
            text = ", ".join([f"{key}: {value}" for key, value in row.items()]) + "."
            documents.append(Document(page_content=text))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chroma_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )
    
    return chroma_db, json_output