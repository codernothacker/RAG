class DocumentProcessingError(Exception):
    """Raised when there's an error processing a document"""
    pass

class VectorStoreError(Exception):
    """Raised when there's an error with the vector store operations"""
    pass

class LLMError(Exception):
    """Raised when there's an error with the LLM operations"""
    pass