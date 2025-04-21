from services.chatbot_service import ChatBotService

# Có thể chuyển thanh get từ database sau đó
chatbot_sessions = {}
chatbot_instance: ChatBotService = None
