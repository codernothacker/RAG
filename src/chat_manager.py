from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.guardrails import EnhancedGuardrails

class ChatManager:
    def __init__(self, vector_store):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.llm = Ollama(model="phi3")
        self.guardrails = EnhancedGuardrails(self.llm)
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.db.as_retriever(),
            memory=self.memory,
            verbose=True
        )
        
        self.logger = logging.getLogger(__name__)

    def get_response(self, query: str) -> str:
        try:
            docs = self.chain.retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])
            initial_response = self.chain({"question": query})
            response = initial_response["answer"]
            filtered_response = self.guardrails.filter_response(
                response=response,
                context=context,
                query=query
            )
            
            if filtered_response != response:
                self.logger.warning(f"Response was modified by guardrails for query: {query}")
            
            return filtered_response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try rephrasing your question."