from langchain.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import ast
import time
import pandas as pd
import os

class ChatBotService:
    def __init__(self, split_memory, conversation_memory, chroma_db, llm, financial_indicators):
        self.split_memory = split_memory
        self.conversation_memory = conversation_memory
        self.chroma_db = chroma_db
        self.llm = llm
        self.financial_indicators = financial_indicators

    def classify_query(self, query):
        """Phân loại câu hỏi là câu hỏi chung hay câu hỏi về dữ liệu công ty"""

        classify_prompt = PromptTemplate.from_template(
            """
            Dựa vào câu hỏi của người dùng, hãy xác định loại câu hỏi là gì:

            Câu hỏi: {query}

            Phân loại câu hỏi này là một trong các loại sau:
            1. "general" - Câu hỏi chung về khái niệm tài chính, kiến thức chung, chào hỏi, hoặc trò chuyện thông thường không yêu cầu dữ liệu cụ thể.
            2. "company_data" - Câu hỏi về số liệu, tình hình tài chính, hoặc thông tin cụ thể của một công ty.

            Trả lời chỉ với "general" hoặc "company_data":
            """
        )

        prompt = classify_prompt.format(query=query)
        response = self.llm.invoke(prompt)
        result = response.content.strip().lower()

        # Đảm bảo kết quả chỉ là "general" hoặc "company_data"
        if "general" in result:
            return "general"
        elif "company_data" in result:
            return "company_data"
        else:
            # Mặc định nếu không xác định được
            return "general"

    def handle_general_query(self, query):
        """Xử lý câu hỏi chung về tài chính, không yêu cầu dữ liệu cụ thể"""

        prompt_template = PromptTemplate.from_template(
            """
            Bạn là một chuyên gia phân tích tài chính cao cấp với nhiều năm kinh nghiệm. Bạn có khả năng hiểu sâu sắc về các chỉ số tài chính, xu hướng thị trường và các yếu tố kinh tế vĩ mô.

            Hãy trả lời câu hỏi sau đây dựa trên kiến thức chung về tài chính và kinh tế của bạn:

            Câu hỏi: {query}

            Lịch sử hội thoại: {chat_history}

            Câu trả lời:
            """
        )

        chat_history = self.conversation_memory.buffer
        prompt = prompt_template.format(query=query, chat_history=chat_history)

        response = self.llm.invoke(prompt)
        content = response.content

        # Lưu vào bộ nhớ hội thoại
        self.conversation_memory.save_context({"input": query}, {"output": content})
        self.split_memory.save_context({"input": query}, {"output": content})

        return content

    def handle_company_data_query(self, query):
        """Xử lý câu hỏi về dữ liệu công ty sử dụng RAG"""
        if not self.chroma_db:
            return "Tôi không có dữ liệu về công ty để trả lời câu hỏi này. Vui lòng cung cấp dữ liệu công ty trước."

        # Split query thành các sub-queries
        sub_queries = self.split_query(query)

        # Tìm kiếm thông tin cho mỗi sub-query
        results_dict = self.search_subqueries_in_chromaDB(sub_queries, k=5)

        # Generate câu trả lời từ kết quả tìm kiếm
        response = self.ask_gemini(query, results_dict)

        return response

    def split_query(self, query):
        prompt_template = PromptTemplate.from_template(
          """
                Bạn là chuyên gia tài chính.
                Lịch sử hội thoại:
                {chat_history}
                Câu hỏi hiện tại: "{query}"
                Các chỉ tiêu tài chính:
                {financial_indicators}
                Kiểu dữ liệu trả về là 1 list trong Python gồm các chuỗi là các query con có dạng:[".....",".......",...]

                Nhiệm vụ của bạn là tách câu hỏi sau thành các câu hỏi nhỏ nếu nó chứa nhiều chỉ tiêu tài chính khác nhau dựa vào câu hỏi hiện tại,lịch sử hội thoại và Các chỉ tiêu tài chính.
                Đặc biệt dựa vào Tên chỉ tiêu trong {financial_indicators} để lấy ra các query con liên quan đến câu hỏi hiện tại là {query}
                Vui lòng trả về kết quả theo định dạng sau:
                - Ví dụ query là:"Chi phí và Thuế cuối năm của công ty mẹ vào năm nay và năm trước là bao nhiêu?" thì kết quả trả về là:
                    ["Chi phí cuối năm của công ty mẹ năm nay","Thuế cuối năm của công ty mẹ năm nay","Chi phí cuối năm của công ty mẹ năm trước","Thuế cuối năm của công ty mẹ năm trước"]
                - Dựa vào {query} hãy đưa ra các query con có thể trả lời được cho query. Ví dụ query như sau:"Công ty lời hay lỗ?" thì phải tự lấy ra các chỉ tiêu giải đáp được như ["Lợi nhuận sau thuế TNDN của công ty", hoặc các chỉ tiêu để tính các thông số mà người dùng hỏi, như hỏi ROA, ROE, Lợi nhuận hoạt động biên,.. thì phải trả về các query con chứa các chỉ tiêu để tính được các thông số đó.
                - Nếu câu hỏi chỉ chứa một chỉ tiêu tài chính thì trả về 1 query gồm chỉ tiêu đó.
                - Vì bạn là một chuyên gia về tài chính nên bạn phải hiểu rõ cấu trúc, các chỉ tiêu của 3 bảng: Bảng cân đối kế toán, Báo cáo KQHĐKD, Bảng Lưu chuyển tiền tệ.
                    Đối với một người mới, họ chỉ biết sơ lược về các chỉ tiêu nên {query} có thể chỉ là Lợi nhuận sau thuế, Nhưng thực chất lợi nhuận sau thuế có rất nhiều loại trong 3 bảng trên, như Lợi nhuận sau thuế của công ty mẹ , Lợi nhuận sau thuế của cổ đông không kiểm soát, Lợi nhuận sau thuế chưa phân phối, Lợi nhuận sau thuế chưa phân phối đến cuối năm trước, Lợi nhuận sau thuế chưa phân phối năm nay
                    Tương tự đối với các chỉ tiêu khác.Trong trường hợp này, kết quả đầu ra nên là các query con đầy đủ các trường hợp.
                    Ví dụ đầu vào là "Chi phí trả trước là bao nhiêu?" thì kết quả trả về là:
                    ["Chi phí trả trước ngắn hạn năm nay","Chi phí trả trước ngắn hạn năm trước","Chi phí trả trước dài hạn năm nay","Chi phí trả trước dài hạn năm trước"]

                PHÂN TÍCH NGỮA CẢNH - CỰC KỲ QUAN TRỌNG:
                1. Nếu câu hỏi hiện tại chứa các cụm từ như "còn", "vậy còn", "thế còn", "thì sao", hoặc chỉ đề cập đến thời gian mà không nêu rõ chỉ tiêu (ví dụ: "đầu năm thì sao", "cuối năm là bao nhiêu"), thì đây là dấu hiệu câu hỏi đang tham chiếu đến các chỉ tiêu tài chính đã được đề cập trong lịch sử hội thoại gần nhất.
                VÍ DỤ NGỮA CẢNH:
                - Nếu lịch sử có "Lợi nhuận sau thuế là bao nhiêu?" và câu hỏi hiện tại là "Cuối năm là bao nhiêu?" thì kết quả trả về là:
                  ["Lợi nhuận sau thuế TNDN","Lợi nhuận sau thuế của công ty mẹ là bao nhiêu?"]
                  Trong trường hợp này, hãy phân tích lịch sử hội thoại và xác định chính xác các chỉ tiêu tài chính mà người dùng đã hỏi trước đó. Đặc biệt chú ý đến câu hỏi và câu trả lời gần nhất.
                  Khi xác định được các chỉ tiêu liên quan từ lịch sử, hãy tạo các câu hỏi con chứa đúng các chỉ tiêu đó, dựa vào AIMessage của lịch sử hội thoại gần nhất.

                Nếu đó là câu hỏi về chỉ tiêu lần đầu tiên thì không cần dựa vào lịch sử chat, chỉ dựa vào câu hỏi hiện tại + Các chỉ tiêu tài chính.
                Vui lòng trả về kết quả theo định dạng sau:
                - Ví dụ query là:"Tổng cộng tài sản và Lợi nhuận năm nay là bao nhiêu?" thì kết quả trả về là:
                    [ "Tổng cộng tài sản năm nay", "Lợi nhuận sau thuế TNDN năm nay","Lợi nhuận sau thuế của công ty mẹ năm nay",........]
                    Hoặc query là:"Chi phí và Thuế cuối năm của công ty mẹ vào năm nay và năm trước là bao nhiêu?" thì kết quả trả về là:
                    ["Chi phí cuối năm của công ty mẹ","Thuế cuối năm của công ty mẹ",.....]
                    ...'''
                -Nếu câu hỏi chỉ chứa một chỉ tiêu tài chính thì trả về 1 query gồm chỉ tiêu đó.
                - Nếu câu hỏi chỉ là lời chào hỏi, không có đề cập đến chỉ tiêu hay hỏi điều gì thì chỉ đơn giản chào lại và giới thiệu bản thân là chuyên gia tài chính.
                - Nếu câu trả lời không đúng định dạng sẽ bị loại bỏ, phải là LIST các query con, đặc biệt KHÔNG ĐƯỢC TRẢ VỀ MÃ PYTHON và không được là CHUỖI String json

                Nếu câu hỏi hiện tại hỏi chỉ tiêu được lấy từ bảng nào thì hãy trả về query con là chỉ tiêu đó. Ví dụ:
                Query hiện tại là "Tổng cộng tài sản là bao nhiêu" --> kết quả trả về là: ["Tổng cộng tài sản cuối năm","Tổng cộng tài sản đầu năm"]
                Query tiếp theo: "Được lấy từ bảng nào" thì phải dựa trên lịch sử hội thoại HumanMessage gần nhất trả về : ["Tổng cộng tài sản cuối năm được lấy từ bảng nào?", "Tổng cộng tài sản đầu năm được lấy từ bảng nào?"]
                Lưu ý: Kết quả trả về là query con dạng câu hỏi con của câu hỏi hiện tại, không phải câu trả lời cho câu hỏi hiện tại.
                - Đặc biệt dù query có dạng như thế nào thì hãy trả về kết quả query con chỉ chứa chỉ tiêu liên quan, không chứa các từ không liên quan như "được tính như thế nào?","bao nhiêu",... sẽ làm nhiễu khi embedding dẫn đến kết quả không chính xác
                Không giải thích gì thêm.
          """)

        chat_history = self.split_memory.buffer
        prompt = prompt_template.format(
            query=query,
            chat_history=chat_history,
            financial_indicators=self.financial_indicators
        )

        response = self.llm.invoke(prompt)
        content = response.content
        clean_content = content.replace("```python\n", "").replace("```", "").strip()
        try:
            queries = ast.literal_eval(clean_content)
        except Exception as e:
            print("Lỗi convert literal:", e)
            print("Nội dung:", clean_content)
            queries = []
        # queries = ast.literal_eval(clean_content)
        self.split_memory.save_context({"input": query}, {"output": "\n".join(queries)})

        return queries

    def search_subqueries_in_chromaDB(self, queries, k):
        retriever = self.chroma_db.as_retriever(search_kwargs={"k": k})
        results_dict = {}

        # Truy vấn retriever cho từng sub-query
        for sub_query in queries:
            retrieved = retriever.invoke(sub_query)
            contents = []

            for r in retrieved:
                if isinstance(r, Document):
                    contents.append(r.page_content)
                else:
                    contents.append(r)
            results_dict[sub_query] = contents

        return results_dict

    def ask_gemini(self, query, results_dict):
        # Format search results into a context string
        full_context = ""
        for q, answers in results_dict.items():
            if answers:
                full_context += f"Câu hỏi: {q}\n"
                for answer in answers:
                    full_context += f"Kết quả: {answer}\n"

        chat_history = self.conversation_memory.buffer
        # Create prompt with context
        prompt_template = PromptTemplate.from_template(
            """
                Bạn là một chuyên gia phân tích tài chính cao cấp với nhiều năm kinh nghiệm. Bạn có khả năng hiểu sâu sắc về các chỉ số tài chính, xu hướng thị trường và các yếu tố kinh tế vĩ mô.
                Đầu tiên dựa vào câu hỏi gốc là {query}, thông tin của câu hỏi gốc là {full_context} và dựa vào lịch sử hội thoại là {chat_history} để đưa các câu trả lời phù hợp để trả lời cho {query}
                Ngoài ra nếu cần các thông tin số liệu của các chỉ tiêu khác để phục vụ cho việc trả lời câu hỏi của người dùng là {query} thì hãy dựa vào {financial_indicators} để bổ sung thông tin số liệu.
                Mỗi {full_context} CÓ THỂ (Tuy xác suất rât nhỏ) nhưng vẫn có thể chứa nhiều thông tin không liên quan hoặc không cần thiết để trả lời cho {query}, hãy lựa chọn và sử dụng những thông tin chính xác để trả lời cho {query}, nếu {query} không hỏi đến số liệu thì không cần dựa vào {full_context} mà hãy trả lời cho dựa vào kiến thức Gemini được train bởi Google.
                Đối với các chỉ tiêu bình quân thì lấy bình quân năm nay cho cả năm nay và năm trước, không cần tính lại, lấy chung 1 số cho cả năm nay và năm trước.
                PHÂN TÍCH NGỮ CẢNH - CỰC KỲ QUAN TRỌNG:
                1. Nếu câu hỏi hiện tại là {query} chứa các cụm từ như "còn", "vậy còn", "thế còn", "thì sao", hoặc chỉ đề cập đến thời gian mà không nêu rõ chỉ tiêu (ví dụ: "đầu năm thì sao", "cuối năm là bao nhiêu") hoặc có dấu hiệu câu hỏi đang tham chiếu đến các chỉ tiêu tài chính đã được đề cập trong lịch sử hội thoại gần nhất {chat_history} để đưa ra câu trả lời logic và liên kết với nhau.

                VÍ DỤ NGỮ CẢNH:
                - Nếu lịch sử có "Lợi nhuận sau thuế đầu năm là bao nhiêu?" và câu hỏi hiện tại là "Cuối năm là bao nhiêu?" thì hãy dựa vào lịch sử cuộc trò chuyện là {chat_history} để đưa ra câu trả lời có tính liên kết, ngôn ngữ tự nhiên

                2. Trong trường hợp này, hãy phân tích lịch sử hội thoại và xác định chính xác các chỉ tiêu tài chính mà người dùng đã hỏi trước đó. Đặc biệt chú ý đến câu hỏi và câu trả lời gần nhất.
                3. Khi xác định được các chỉ tiêu liên quan từ lịch sử, hãy tạo các câu hỏi con chứa đúng các chỉ tiêu đó
                4. Đối với những query đơn thuần là chào hỏi, chưa hỏi gì về tài chính thì chỉ chào hỏi lại, giới thiệu bản thân là chuyên gia tài chính và trả lời thắc mắc từ người dùng chứ đừng trích xuất bất cứ thông tin nào mà người dùng không hỏi.
                5. Người dùng hỏi gì thì trả lời đó, tuyệt đối không có tình trạng người dùng không hỏi mà trả về hàng loạt thông tin số liệu trong khi chưa hỏi.
                Kết quả trả về phải tuân thủ các quy định sau:

                - Nếu kết quả trong context không phải là câu trả lời phù hợp cho {query} thì bỏ qua và không trả về vì không trả lời đúng trọng tâm.
                - Nếu không tìm thấy dữ liệu cho một số trường, hãy nêu rõ rằng thông tin không có sẵn thay vì suy đoán.
                - Chỉ dựa vào dữ liệu đã được cung cấp, không thêm thông tin không có trong nguồn.
                - Trả lời ngắn gọn nhưng đầy đủ, sử dụng phong cách chuyên nghiệp và dễ hiểu cho nhà đầu tư và lãnh đạo doanh nghiệp.
                - Đối với những câu hỏi vì sao thì phải dựa trên kiến thức tài chính + số liệu có được để trả lời 1 cách logic và thuyết phục
                - Lưu ý đầu ra là văn bản thuần túy, không chứa các ký tự đặc biệt như *, /,... Trình bày rõ ràng bằng cách xuống dòng hoặc chia đoạn hợp lý.
                - Trả lời tự nhiên, không quá máy móc.

                Các cuộc trao đổi trước đây: {chat_history}
            """
        )

        prompt = prompt_template.format(
            query=query,
            full_context=full_context,
            chat_history=chat_history,
            financial_indicators=self.financial_indicators
        )

        response = self.llm.invoke(prompt)
        content = response.content

        # Lưu vào bộ nhớ hội thoại
        self.conversation_memory.save_context(
            {"input": query},
            {"output": content}
        )

        return content

    def process_query(self, query):
        """Phương thức chính để xử lý câu hỏi từ người dùng"""
        # Bước 1: Phân loại câu hỏi
        query_type = self.classify_query(query)

        # Bước 2: Xử lý dựa trên loại câu hỏi
        if query_type == "general":
            return self.handle_general_query(query)
        else:  # query_type == "company_data"
            return self.handle_company_data_query(query)

#Tạo agent
def create_finance_agent(chatbot_api_key, dataframe=None):
    if dataframe is not None:
        dict_json = ChatBotService.convert_df_to_dict_json(dataframe)
        agent = ChatBotService(chatbot_api_key, dict_json)
    else:
        agent = ChatBotService(chatbot_api_key)

    return agent
