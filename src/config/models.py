from abc import ABC

from langchain_groq import ChatGroq

class Models(ABC):
    def __init__(self):
        self.chat_model = ChatGroq(model="llama3-70b-8192")
        self.json_model = self.chat_model.bind(response_format={"type": "json_object"})
        
        # High Token model (higher limit for tokens per request, but has daily limit)
        self.ht_model = ChatGroq(model="llama-3.1-70b-versatile")
        self.ht_json_model = self.ht_model.bind(response_format={"type": "json_object"})
