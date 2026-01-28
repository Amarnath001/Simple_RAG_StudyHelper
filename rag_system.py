"""
RAG (Retrieval-Augmented Generation) system for study assistance.
"""
import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class LocalEmbeddings:
    """Wrapper for sentence-transformers to work with LangChain."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        # Lazy import to avoid dependency issues at module level
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class RAGSystem:
    """RAG system for querying study materials and generating responses."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system.
        
        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = persist_directory
        try:
            # Use local embeddings (free, no API calls needed)
            # Using a lightweight, fast embedding model
            self.embeddings = LocalEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            # Use Gemini model for text generation
            # Allow model name to be configured via environment variable
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
            # gemini-1.5-flash is the most commonly available free model
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.7,
                convert_system_message_to_human=True
            )
        except Exception as e:
            error_msg = str(e)
            # Embeddings are local, so errors here are about the LLM
            if "api_key" in error_msg.lower() or "api key" in error_msg.lower():
                raise Exception(
                    "Gemini API key not found. Please set GEMINI_API_KEY in your .env file. "
                    "Get your API key from: https://makersuite.google.com/app/apikey"
                )
            if "quota" in error_msg.lower() or "429" in error_msg.lower():
                raise Exception(
                    "Gemini API quota exceeded. Please check your usage at "
                    "https://makersuite.google.com/app/apikey or wait for quota reset."
                )
            if "not found" in error_msg.lower() or "404" in error_msg.lower():
                raise Exception(
                    f"Gemini model not found. Error: {error_msg}\n\n"
                    "Try checking available models at: https://makersuite.google.com/app/apikey\n"
                    "Common model names: gemini-1.5-flash, gemini-1.5-pro, gemini-pro"
                )
            raise
        self.vectorstore = None
        
    def initialize_vectorstore(self, documents: List[Document]):
        """
        Initialize or update the vector store with documents.
        
        Args:
            documents: List of Document objects to add to the vector store
        """
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            # Load existing vector store
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            # Add new documents
            self.vectorstore.add_documents(documents)
        else:
            # Create new vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
    def query(self, question: str) -> dict:
        """
        Query the RAG system with a question.
        
        Args:
            question: Student's question
            
        Returns:
            Dictionary with 'answer' and 'source_documents'
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""You are a helpful study assistant. Use the following pieces of context from study materials to answer the student's question. 
        If you don't know the answer based on the context, say so. Be clear, educational, and helpful.

        Context: {context}

        Question: {question}

        Answer: Provide a clear, educational answer based on the context. If the answer isn't in the context, say so."""
        
        # Get answer from LLM
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "source_documents": docs
        }
    
    def generate_tutorial(self, topic: str) -> str:
        """
        Generate a tutorial on a specific topic using the study materials.
        
        Args:
            topic: Topic for the tutorial
            
        Returns:
            Generated tutorial text
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(topic, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        tutorial_prompt = f"""Based on the following study materials, create a comprehensive tutorial on the topic: {topic}

        Study Materials:
        {context}

        Create a tutorial that:
        1. Introduces the topic clearly
        2. Explains key concepts step by step
        3. Provides examples where relevant
        4. Summarizes the main points

        Tutorial:"""
        
        response = self.llm.invoke(tutorial_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def generate_questions(self, topic: Optional[str] = None, num_questions: int = 5) -> List[str]:
        """
        Generate practice questions based on the study materials.
        
        Args:
            topic: Optional specific topic to focus on
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        # Retrieve relevant documents
        if topic:
            docs = self.vectorstore.similarity_search(topic, k=5)
        else:
            docs = self.vectorstore.similarity_search("main concepts", k=5)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        question_prompt = f"""Based on the following study materials, generate {num_questions} practice questions that would help students test their understanding.

        Study Materials:
        {context}

        Generate questions that:
        1. Test understanding of key concepts
        2. Range from basic to more advanced
        3. Are clear and specific
        4. Cover different aspects of the material

        Questions (one per line, numbered):"""
        
        response = self.llm.invoke(question_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        # Parse questions from response
        questions = [q.strip() for q in response_text.split('\n') if q.strip() and (q.strip()[0].isdigit() or q.strip().startswith('-') or q.strip().startswith('•'))]
        return questions[:num_questions] if questions else [response_text]
    
    def generate_study_notes(self, topic: Optional[str] = None) -> str:
        """
        Generate study notes/summary for a topic.
        
        Args:
            topic: Optional specific topic to focus on
            
        Returns:
            Generated study notes
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        # Retrieve relevant documents
        if topic:
            docs = self.vectorstore.similarity_search(topic, k=5)
        else:
            docs = self.vectorstore.similarity_search("summary overview", k=5)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        notes_prompt = f"""Based on the following study materials, create concise study notes that summarize key points.

        Study Materials:
        {context}

        Create study notes that:
        1. Highlight key concepts and definitions
        2. Organize information clearly
        3. Include important formulas or facts
        4. Are easy to review

        Study Notes:"""
        
        response = self.llm.invoke(notes_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_similar_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Get similar documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        return self.vectorstore.similarity_search(query, k=k)
