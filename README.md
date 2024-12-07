# LangChain RAG System

A Retrieval Augmented Generation (RAG) system built with LangChain, ChromaDB, and Ollama that supports both structured and unstructured documents with a Streamlit interface.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Component Details](#component-details)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## Features
- Multi-format document processing (PDF, TXT, CSV, JSON, XLSX)
- Vector-based document retrieval
- Local LLM inference using Ollama
- Conversational memory
- Response filtering with guardrails
- Web-based user interface

## Project Structure
```
langchain_rag/
├── src/
│   ├── document_loader.py    # Document processing
│   ├── vector_store.py       # Vector database management
│   ├── chat_manager.py       # Chat handling and memory
│   ├── app.py               # Streamlit interface
│   └── utils/
│       ├── logging_config.py # Logging setup
│       ├── prompts.py       # LangChain prompts
│       └── guardrails.py    # Response filtering
```

## Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install unstructured[all-docs]
```

3. Install Ollama:
- Download from https://ollama.ai
- Pull the model:
```bash
ollama pull phi3
```

## Component Details

### DocumentLoader (document_loader.py)

#### Class: `DocumentLoader`
Main class for handling document processing.

Methods:
- `__init__()`: Initializes text splitter with chunk settings
- `get_file_extension(file_path)`: Extracts and validates file extension
- `load_document(file_path)`: Processes documents based on file type

Supported formats:
- PDF: Uses PyPDFLoader
- TXT: Uses TextLoader with UTF-8 encoding
- CSV: Uses CSVLoader with custom delimiter settings
- JSON: Custom implementation for JSON structure
- XLSX/XLS: Uses UnstructuredExcelLoader

### VectorStore (vector_store.py)

#### Class: `VectorStore`
Manages document embeddings and similarity search.

Methods:
- `__init__()`: Initializes ChromaDB with HuggingFace embeddings
- `add_documents(documents)`: Adds documents to vector store
- `similarity_search(query, k=4)`: Retrieves similar documents

### ChatManager (chat_manager.py)

#### Class: `ChatManager`
Handles conversation flow and memory.

Methods:
- `__init__(vector_store)`: Sets up conversation memory and LLM
- `get_response(query)`: Generates responses using context and history

### Streamlit App (app.py)

#### Function: `save_uploaded_file(uploaded_file)`
Saves uploaded files to temporary storage.

#### Function: `main()`
Main application loop:
- Initializes session state
- Handles file uploads
- Manages chat interface
- Processes user queries

### Utils

#### logging_config.py
Configures logging with file and stream handlers.

#### prompts.py
Defines LangChain prompt templates for RAG.

#### guardrails.py
Implements response filtering and validation.

## Usage Guide

1. Start Ollama server:
```bash
ollama serve
```

2. Launch the application:
```bash
streamlit run src/app.py
```

3. Upload documents:
- Use the file upload widget
- Supported formats: PDF, TXT, CSV, JSON, XLSX
- Wait for processing confirmation

4. Chat interaction:
- Type questions in the chat input
- System retrieves relevant context
- Responses are generated using local LLM

## API Reference

### Document Processing
```python
# Load and process document
documents = document_loader.load_document(file_path)

# Document format
{
    'page_content': str,  # Actual content
    'metadata': {         # Metadata
        'source': str,    # File path
        'page': int,      # Page number (if applicable)
    }
}
```

### Vector Store Operations
```python
# Add documents to store
vector_store.add_documents(documents)

# Search similar documents
results = vector_store.similarity_search(query, k=4)
```

### Chat Operations
```python
# Get response for query
response = chat_manager.get_response(query)
```

## Troubleshooting

Common Issues:

1. ModuleNotFoundError: pwd
   - Solution: Install Windows-compatible packages
   ```bash
   pip install python-magic-win64
   ```

2. Torch Class Error:
   - Solution: Upgrade dependencies
   ```bash
   pip install --upgrade langchain langchain-community
   ```

3. File Processing Errors:
   - Check file encoding (use UTF-8)
   - Verify file permissions
   - Ensure temp directory exists

4. Memory Issues:
   - Reduce chunk size in DocumentLoader
   - Decrease max_results in similarity search

For additional issues, check logs in `logs/app.log`.

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request with tests

## License

MIT License - see LICENSE file for details.