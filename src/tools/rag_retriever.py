import glob
import pickle
import qdrant_client

from abc import ABC

from langchain_groq import ChatGroq

from llama_parse import LlamaParse, ResultType

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.config.env import settings


class RAGRetriever(ABC):
    def __init__(self, chat_model: ChatGroq):
        pdf_list = [f.split('/')[-1] for f in glob.glob('rag_source/*.pdf')]

        try:
            with open('rag_source/metadata.pkl', 'rb') as handle:
                old_pdf_list = pickle.load(handle)
            update_collection = False if old_pdf_list == pdf_list else True
            print('No updates detected on the RAG source files, proceeding with current collection.')
        except:
            update_collection = True
            print('RAG source files changed, updating collection.')

        client = qdrant_client.QdrantClient(api_key=settings.QDRANT_API_KEY, url=settings.QDRANT_URL)
        modelPath = "sentence-transformers/all-MiniLM-l6-v2"
        embedding_model = HuggingFaceEmbedding(model_name=modelPath)
        
        Settings.embed_model = embedding_model
        Settings.llm = chat_model
            
        if update_collection:
            pdf_files = glob.glob('rag_source/*.pdf')
            parsed_documents = []
            for pdf_file in pdf_files:
                parsed_documents.extend(LlamaParse(result_type=ResultType.MD).load_data(pdf_file))
            
            vector_store = QdrantVectorStore(client=client, collection_name='pdf_paper_rag')
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents=parsed_documents, storage_context=storage_context, show_progress=True)
            
            with open('rag_source/metadata.pkl', 'wb') as handle:
                pickle.dump(pdf_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vector_store = QdrantVectorStore(client=client, collection_name='pdf_paper_rag')
            index = VectorStoreIndex.from_vector_store(vector_store, embedding_model)
        
        self.query_engine = index.as_query_engine()
    
    def execute(self, query):
        return self.query_engine.query(query)