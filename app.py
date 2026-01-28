"""
Streamlit app for the RAG Study Helper.
"""
import streamlit as st
import os
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Study Helper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()

def load_documents(uploaded_files):
    """Load and process uploaded documents."""
    if not uploaded_files:
        return False
    
    try:
        all_documents = []
        
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                # Process file
                text = st.session_state.doc_processor.process_uploaded_file(uploaded_file)
                
                # Chunk the text
                metadata = {"source": uploaded_file.name}
                chunks = st.session_state.doc_processor.chunk_text(text, metadata)
                all_documents.extend(chunks)
            
            # Initialize RAG system
            if st.session_state.rag_system is None:
                st.session_state.rag_system = RAGSystem()
            
            # Add documents to vector store
            st.session_state.rag_system.initialize_vectorstore(all_documents)
            st.session_state.documents_loaded = True
            
            st.success(f"Successfully loaded {len(uploaded_files)} file(s) with {len(all_documents)} chunks!")
            return True
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return False

def main():
    """Main application function."""
    st.title("📚 RAG Study Helper")
    st.markdown("### Your AI-powered study assistant for learning, tutorials, and practice questions")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ OPENAI_API_KEY not found in environment variables!")
            st.info("Please create a .env file with your OPENAI_API_KEY")
            st.stop()
        else:
            st.success("✅ API Key configured")
        
        # Document upload
        uploaded_files = st.file_uploader(
            "Upload Study Materials",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or MD files containing your study materials"
        )
        
        if st.button("Load Documents", type="primary"):
            if uploaded_files:
                load_documents(uploaded_files)
            else:
                st.warning("Please upload at least one file")
        
        # Status
        st.divider()
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded and ready!")
        else:
            st.info("📤 Upload documents to get started")
        
        # Clear button
        if st.button("Clear All Documents"):
            st.session_state.rag_system = None
            st.session_state.documents_loaded = False
            st.rerun()
    
    # Main content area
    if not st.session_state.documents_loaded:
        st.info("👈 Please upload and load your study materials from the sidebar to begin!")
        st.markdown("""
        ### Features:
        - **Q&A**: Ask questions about your study materials
        - **Tutorials**: Generate step-by-step tutorials on any topic
        - **Practice Questions**: Get practice questions to test your understanding
        - **Study Notes**: Generate concise study notes and summaries
        """)
    else:
        # Feature tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Q&A", "📖 Tutorials", "❓ Practice Questions", "📝 Study Notes"])
        
        # Q&A Tab
        with tab1:
            st.header("Ask Questions")
            st.markdown("Ask any question about your study materials and get AI-powered answers!")
            
            question = st.text_input("Enter your question:", placeholder="e.g., What is the main concept in chapter 3?")
            
            if st.button("Get Answer", type="primary") and question:
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.rag_system.query(question)
                        
                        st.markdown("### Answer:")
                        st.write(result["answer"])
                        
                        # Show source documents
                        with st.expander("View Source Documents"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Tutorials Tab
        with tab2:
            st.header("Generate Tutorials")
            st.markdown("Get step-by-step tutorials on any topic from your study materials!")
            
            tutorial_topic = st.text_input("Enter a topic for the tutorial:", placeholder="e.g., Machine Learning Basics")
            
            if st.button("Generate Tutorial", type="primary") and tutorial_topic:
                with st.spinner("Generating tutorial..."):
                    try:
                        tutorial = st.session_state.rag_system.generate_tutorial(tutorial_topic)
                        st.markdown("### Tutorial:")
                        st.write(tutorial)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Practice Questions Tab
        with tab3:
            st.header("Practice Questions")
            st.markdown("Generate practice questions to test your understanding!")
            
            col1, col2 = st.columns(2)
            with col1:
                question_topic = st.text_input("Topic (optional):", placeholder="Leave empty for general questions")
            with col2:
                num_questions = st.number_input("Number of questions:", min_value=1, max_value=10, value=5)
            
            if st.button("Generate Questions", type="primary"):
                with st.spinner("Generating questions..."):
                    try:
                        questions = st.session_state.rag_system.generate_questions(
                            topic=question_topic if question_topic else None,
                            num_questions=num_questions
                        )
                        st.markdown("### Practice Questions:")
                        for i, q in enumerate(questions, 1):
                            st.markdown(f"**{i}.** {q}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Study Notes Tab
        with tab4:
            st.header("Study Notes")
            st.markdown("Generate concise study notes and summaries!")
            
            notes_topic = st.text_input("Topic for study notes (optional):", placeholder="Leave empty for general notes")
            
            if st.button("Generate Study Notes", type="primary"):
                with st.spinner("Generating study notes..."):
                    try:
                        notes = st.session_state.rag_system.generate_study_notes(
                            topic=notes_topic if notes_topic else None
                        )
                        st.markdown("### Study Notes:")
                        st.write(notes)
                        
                        # Download button
                        st.download_button(
                            label="Download Notes",
                            data=notes,
                            file_name="study_notes.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
