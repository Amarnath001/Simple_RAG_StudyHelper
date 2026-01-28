"""
Document processing utilities for loading and chunking study materials.
"""
import os
from typing import List
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles loading and processing of study materials."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, file_path: str) -> str:
        """
        Load text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def load_text_file(self, file_path: str) -> str:
        """
        Load text from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error loading text file: {str(e)}")
    
    def process_uploaded_file(self, uploaded_file) -> str:
        """
        Process an uploaded file from Streamlit.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Extracted text content
        """
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Determine file type and load accordingly
            if uploaded_file.name.endswith('.pdf'):
                text = self.load_pdf(temp_path)
            elif uploaded_file.name.endswith(('.txt', '.md')):
                text = self.load_text_file(temp_path)
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.name}")
            
            return text
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Document objects
        """
        if metadata is None:
            metadata = {}
        
        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
        
        return documents
