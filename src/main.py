import streamlit as st
from pathlib import Path
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.chat_manager import ChatManager
from src.utils.logging_config import setup_logging
from src.config import settings

def initialize_app():
    logger = setup_logging()
    Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    Path("data/uploaded").mkdir(parents=True, exist_ok=True)
    vector_store = VectorStore()
    chat_manager = ChatManager(vector_store)
    document_loader = DocumentLoader()
    return logger, vector_store, chat_manager, document_loader