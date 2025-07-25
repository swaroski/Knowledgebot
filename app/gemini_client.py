import google.generativeai as genai
from typing import List, Dict, Any
from .config import config

class GeminiClient:
    def __init__(self):
        config.validate()
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.embedding_model = "models/embedding-001"
        self.chat_model = genai.GenerativeModel("gemini-pro")
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using Gemini embedding model"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []
    
    def ask_question(self, question: str, context: str, conversation_history: str = "") -> str:
        """Ask question with context using Gemini chat model"""
        try:
            prompt = f"""You are a helpful Q&A assistant. Use the provided context to answer the user's question accurately and concisely.

Context from documents:
{context}

Conversation History:
{conversation_history}

User Question: {question}

Instructions:
- Base your answer primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Provide specific details and examples when available
- Keep your response focused and relevant
- If referencing specific information, mention which document it came from

Answer:"""

            response = self.chat_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

gemini_client = GeminiClient()