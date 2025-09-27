from abc import ABC

from langchain_groq import ChatGroq
from src.config.env import settings


class Models(ABC):
    def __init__(self):
        self.chat_model = ChatGroq(model="llama-3.1-8b-instant", api_key=settings.GROQ_API_KEY)
        self.json_model = self.chat_model.bind(response_format={"type": "json_object"})
        
        # High Token model (higher limit for tokens per request, but has daily limit)
        self.ht_model = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)
        self.ht_json_model = self.ht_model.bind(response_format={"type": "json_object"})
