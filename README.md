# YouTube RAG

A Retrieval-Augmented Generation (RAG) application for YouTube videos with enhanced retrieval accuracy.

## Features

- Process YouTube videos to extract transcripts (using YouTube API or Whisper speech-to-text)
- Create semantic search indexes for video content
- Ask questions about video content with precise timestamp references
- Generate summaries of videos
- Click on timestamps in answers to jump to specific parts of videos
- Two-stage retrieval system with cross-encoder reranking for better accuracy
- Efficient chunking with overlap for context preservation

## Requirements

- Python 3.8+
- Ollama for local LLM inference
- FFmpeg for audio processing (when using Whisper)
- ChromaDB for vector storage
- Sentence Transformers for embeddings and reranking

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start Ollama on your system (see https://ollama.ai)
4. Pull a model in Ollama:
   ```
   ollama pull llama3.2
   ```
5. Copy `.env.example` to `.env` and configure as needed
6. Start the application:
   ```
   python app.py
   ```

## Configuration

Key configuration options in the `.env` file:

```
# Ollama configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT=30

# Debug mode (optional)
DEBUG=False
```

## API Endpoints

### Core Endpoints

- `POST /process_video` - Process a YouTube video
  - Extracts transcript using YouTube API or Whisper
  - Splits content into chunks with overlap
  - Stores chunks in vector database

- `POST /ask` - Ask a question about a video
  - Uses two-stage retrieval (vector search + reranking)
  - Returns answer with clickable timestamps
  - Provides context from most relevant chunks

- `POST /summarize` - Generate a summary of a video
  - Creates concise video summary
  - Caches results for efficiency
  - Includes key points and timestamps

- `GET /get_processed_videos` - List all processed videos
- `GET /get_chunks` - Get transcript chunks for a video
- `POST /delete_video` - Delete a processed video
- `GET /config` - Get current configuration
- `GET /get_available_models` - List available Ollama models

## How the Enhanced RAG System Works

1. **Content Processing**:
   - Extract transcripts from YouTube videos
   - Split content into overlapping chunks (500 chars with 100 char overlap)
   - Store chunks with metadata in ChromaDB

2. **Question Answering**:
   - Initial vector search to find relevant chunks
   - Cross-encoder reranking for better accuracy
   - Select top 5 most relevant chunks
   - Generate answer using local LLM
   - Provide timestamps for video navigation

3. **Key Improvements**:
   - Two-stage retrieval system for better accuracy
   - Efficient chunking with overlap for context preservation
   - Local LLM inference for privacy and control
   - Cached summaries for better performance
   - Comprehensive error handling and fallbacks
