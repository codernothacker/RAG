from langchain.prompts import PromptTemplate

RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant. Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Chat History:
{chat_history}
"""