import streamlit as st
import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from ingest_pdf import process_pdf, create_search_index, search_similar
from gemini_helper import init_gemini, generate_explanation, enhance_answer_with_gemini

# Set page config
st.set_page_config(
    page_title="PDF Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {font-size: 24px; color: #1f77b4; margin-bottom: 20px;}
    .result-box {padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 5px solid #1f77b4; background-color: #f8f9fa;}
    .page-info {color: #666; font-size: 0.9em; margin-bottom: 10px;}
    .score {color: #28a745; font-weight: bold;}
    .stButton>button {background-color: #1f77b4; color: white;}
    .stTextInput>div>div>input {border: 1px solid #1f77b4;}
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'search_ready' not in st.session_state:
    st.session_state.search_ready = False
    st.session_state.model = None
    st.session_state.index = None
    st.session_state.metadata = []
    st.session_state.pdf_processed = False
    st.session_state.gemini_model = None
    st.session_state.gemini_initialized = False

def init_gemini_model(api_key: str):
    """Initialize the Gemini model with API key."""
    try:
        if not api_key:
            st.warning("Please enter your Gemini API key to enable AI-powered explanations.")
            return False
            
        with st.spinner("Initializing AI assistant..."):
            st.session_state.gemini_model = init_gemini(api_key)
            st.session_state.gemini_initialized = True
            st.success("AI assistant ready!")
            return True
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return False

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location."""
    try:
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def process_document(uploaded_file=None, file_path=None):
    """Process the PDF document and create search index."""
    if not uploaded_file and not file_path:
        st.error("No file provided for processing")
        return False
        
    with st.spinner('Processing PDF and creating search index... This may take a few minutes...'):
        try:
            # Save uploaded file if provided
            if uploaded_file:
                file_path = save_uploaded_file(uploaded_file)
                if not file_path:
                    return False
            
            # Set the PDF path for processing
            os.environ['PDF_PATH'] = file_path
            
            # Process PDF
            docs = process_pdf()
            
            # Create search index
            index, metadata = create_search_index(docs)
            
            # Load the model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Update session state
            st.session_state.model = model
            st.session_state.index = index
            st.session_state.metadata = metadata
            st.session_state.search_ready = True
            st.session_state.pdf_processed = True
            st.session_state.current_pdf = file_path
            
            st.success("Document processed and ready for search!")
            return True
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return False

def main():
    st.title("📚 PDF Search Engine")
    st.markdown("---")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # File Upload Section
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        # Document Processing
        if not st.session_state.pdf_processed:
            if uploaded_file and st.button("Process PDF", help="Click to process the uploaded PDF"):
                if process_document(uploaded_file=uploaded_file):
                    st.rerun()
        else:
            current_file = os.path.basename(st.session_state.get('current_pdf', 'the document'))
            st.success(f"✅ {current_file} is ready for search")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Reprocess Current"):
                    process_document(file_path=st.session_state.current_pdf)
                    st.rerun()
            with col2:
                if st.button("Upload New"):
                    st.session_state.search_ready = False
                    st.session_state.pdf_processed = False
                    st.rerun()
        
        # AI Assistant Settings
        st.subheader("AI Assistant")
        api_key = st.text_input("Gemini API Key", type="password", 
                              help="Enter your Google Gemini API key for enhanced explanations")
        
        if api_key and not st.session_state.gemini_initialized:
            if st.button("Initialize AI Assistant"):
                init_gemini_model(api_key)
        
        if st.session_state.gemini_initialized:
            st.success("🤖 AI Assistant Active")
        else:
            st.info("ℹ️ Add your Gemini API key for AI-powered explanations")
    
    # Main search interface
    st.markdown('<div class="main-header">Search Your Document</div>', unsafe_allow_html=True)
    
    if not st.session_state.search_ready:
        st.info("Please process the PDF document first using the sidebar.")
        return
    
    # Search box
    query = st.text_input("Enter your search query:", placeholder="Search for information in the document...", key="search_query")
    
    if query:
        with st.spinner('Searching...'):
            try:
                # Perform search and get structured response
                response = search_similar(
                    query, 
                    st.session_state.index, 
                    st.session_state.metadata, 
                    st.session_state.model,
                    k=5  # Number of results to use for context
                )
                
                # Enhance answer with Gemini if available
                answer = response['answer']
                
                if st.session_state.gemini_initialized:
                    with st.spinner("Enhancing answer with AI..."):
                        answer = enhance_answer_with_gemini(
                            query,
                            answer,
                            st.session_state.gemini_model
                        )
                
                # Display the answer
                st.markdown("### Answer")
                st.markdown(answer)
                
                # Show sources if available
                if response.get('sources'):
                    sources_text = ", ".join([f"Page {p}" for p in response['sources']])
                    st.caption(f"📚 Source: {sources_text}")
                    
                    # If we have sources but the answer is short, try to get more context
                    if len(answer) < 200 and st.session_state.gemini_initialized:
                        with st.spinner("Looking for more details..."):
                            context = "\n\n".join([
                                f"Page {r.get('page', '?')}: {r['text']}" 
                                for r in results[:3]  # Get top 3 results for context
                            ])
                            enhanced = generate_explanation(query, context, st.session_state.gemini_model)
                            if enhanced and len(enhanced) > len(answer) + 50:  # Only replace if significantly better
                                st.markdown("### Additional Context")
                                st.markdown(enhanced)
                
                # Display related images if available
                if response.get('images'):
                    st.markdown("### Related Images")
                    
                    # Create columns for image display (2 images per row)
                    cols = st.columns(2)
                    for i, img_path in enumerate(response['images']):
                        if os.path.exists(img_path):
                            with cols[i % 2]:
                                try:
                                    st.image(
                                        img_path,
                                        use_column_width=True,
                                        caption=f"Figure {i+1}"
                                    )
                                except Exception as e:
                                    st.error(f"Could not display image: {str(e)}")
                
                # Add a feedback section
                st.markdown("---")
                st.markdown("### Was this helpful?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Yes"):
                        st.success("Thanks for your feedback!")
                with col2:
                    if st.button("👎 No"):
                        st.text_input("What information were you looking for?", key=f"feedback_{query}")
                        if st.session_state.get(f"feedback_{query}"):
                            st.info("Thank you for your feedback! We'll use this to improve our answers.")
                            
            except Exception as e:
                st.error(f"Error performing search: {str(e)}")

if __name__ == "__main__":
    main()
