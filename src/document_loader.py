from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Optional
import json
import logging
import os

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_file_extension(self, file_path: str) -> Optional[str]:
        try:
            extension = Path(file_path).suffix.lower()
            if not extension:
                self.logger.error(f"No file extension found for {file_path}")
                return None
            return extension
        except Exception as e:
            self.logger.error(f"Error getting file extension: {str(e)}")
            return None

    def load_document(self, file_path: str) -> List:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        extension = self.get_file_extension(file_path)
        if not extension:
            raise ValueError("Invalid file path or missing extension")
        self.logger.info(f"Loading document: {file_path} with extension: {extension}")
        try:
            documents = []
            if extension == '.pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            elif extension == '.csv':
                loader = CSVLoader(
                    file_path,
                    encoding='utf-8',
                    csv_args={
                        'delimiter': ',',
                        'quotechar': '"',
                    }
                )
                documents = loader.load()
                
            elif extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, (list, dict)):
                    text_content = json.dumps(data, indent=2)
                    documents = [{"page_content": text_content, "metadata": {"source": file_path}}]
                
            elif extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(file_path, mode="elements")
                documents = loader.load()
                
            else:
                raise ValueError(f"Unsupported file format: {extension}")

            if not documents:
                raise ValueError(f"No content extracted from {file_path}")
            
            self.logger.info(f"Splitting documents into chunks...")
            split_docs = self.text_splitter.split_documents(documents)
            
            self.logger.info(f"Successfully processed {file_path} into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            raise