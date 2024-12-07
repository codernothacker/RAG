import streamlit as st
from document_loader import DocumentLoader
from vector_store import VectorStore
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chat_manager import ChatManager
import tempfile
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file):
    try:
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return str(temp_path)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise

def main():
    st.title("RAG Chat System")
    
    if "chat_manager" not in st.session_state:
        vector_store = VectorStore()
        st.session_state.chat_manager = ChatManager(vector_store)
        st.session_state.document_loader = DocumentLoader()
        st.session_state.messages = []
        st.session_state.processed_files = set()
    uploaded_file = st.file_uploader(
        "Upload your document", 
        type=["pdf", "txt", "csv", "json", "xlsx", "xls"]
    )
    
    if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                file_path = save_uploaded_file(uploaded_file)
                
                documents = st.session_state.document_loader.load_document(file_path)
                
                if documents:
                    st.session_state.chat_manager.chain.retriever.vectorstore.add_documents(documents)
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.success(f"Successfully processed {uploaded_file.name}")
                else:
                    st.error("No content could be extracted from the file")
                os.remove(file_path)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            
    st.markdown("### Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if query := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        try:
            with st.spinner("Generating response..."):
                response = st.session_state.chat_manager.get_response(query)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()