from typing import Dict, List, Any
from datetime import datetime

class ConversationMemory:
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history = 10  # Keep last 10 exchanges per session
    
    def add_exchange(self, session_id: str, question: str, answer: str, sources: List[Dict] = None):
        """Add a question-answer exchange to session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources or []
        }
        
        self.sessions[session_id].append(exchange)
        
        # Keep only the most recent exchanges
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
    
    def get_conversation_history(self, session_id: str) -> str:
        """Get formatted conversation history for context"""
        if session_id not in self.sessions:
            return ""
        
        history_parts = []
        for exchange in self.sessions[session_id][-5:]:  # Last 5 exchanges for context
            history_parts.append(f"Q: {exchange['question']}")
            history_parts.append(f"A: {exchange['answer'][:200]}...")  # Truncate long answers
        
        return "\n".join(history_parts)
    
    def get_session_exchanges(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all exchanges for a session"""
        return self.sessions.get(session_id, [])
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_all_sessions(self) -> List[str]:
        """Get all session IDs"""
        return list(self.sessions.keys())

conversation_memory = ConversationMemory()