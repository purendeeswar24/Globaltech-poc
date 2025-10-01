# AI-Powered Document Search with Gemini Integration

A powerful document processing and search system that extracts educational content from PDFs, creates searchable indexes, and provides AI-powered answers using Google's Gemini model.

## Features

- **Smart PDF Processing**: Extracts text and educational diagrams from PDFs
- **semantic Search**: Finds relevant content using vector embeddings
- **AI-Powered Answers**: Generates human-like responses using Google's Gemini
- **Image Extraction**: Identifies and extracts educational diagrams and figures
- **Interactive Web UI**: User-friendly interface built with Streamlit

## Technical Stack

- **PDF Processing**: `pdfplumber`, `Pillow`
- **NLP & Embeddings**: `sentence-transformers`
- **Vector Search**: `faiss-cpu`
- **Web Interface**: `streamlit`
- **AI Model**: Google Gemini API

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Google Gemini API key (for AI features)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Img_Chat/Neet
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your browser to `http://localhost:8501`

3. Use the sidebar to:
   - Upload a PDF document
   - Process the document (first-time processing may take a few minutes)
   - Enter your Gemini API key for AI features

4. Search for information and get instant answers!

## How It Works

### 1. Document Processing Pipeline
- Extracts text and processes PDF pages
- Identifies and saves educational diagrams and figures
- Chunks text into manageable segments with overlap
- Creates vector embeddings for semantic search

### 2. Search & Retrieval
- Converts queries into vector embeddings
- Uses FAISS for efficient similarity search
- Ranks results by relevance
- Retrieves context from multiple document sections

### 3. AI Enhancement
- Uses Google's Gemini model to generate human-like answers
- Provides context-aware responses
- Explains complex concepts using document content

## Project Structure

```
Neet/
├── .env                    # Environment variables
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── streamlit_app.py        # Main Streamlit application
├── ingest_pdf.py          # PDF processing and search functionality
├── gemini_helper.py       # Google Gemini integration
└── output/                # Processed data and extracted images
    └── images/            # Extracted educational diagrams
```

## AI Integration

This project uses Google's Gemini model for enhanced question answering. To enable AI features:

1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/)
2. Add it to your `.env` file
3. Enter it in the sidebar of the web interface

## Notes

- First-time PDF processing may take several minutes depending on document size
- Works best with well-structured educational PDFs
- Optimized for documents with clear section headings and figure captions

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [Google Gemini](https://ai.google.dev/) for AI-powered question answering