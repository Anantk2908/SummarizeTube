# YouTube RAG with Ollama

A web application that processes YouTube videos, extracts transcripts, and uses Ollama LLMs for question-answering and summarization.

## Features

- Extract transcripts with timestamps from YouTube videos using YouTube's API or Whisper
- Ask questions about video content using RAG with timestamp references
- Generate summaries of video content
- Support for multiple Ollama models

## Prerequisites

- Python 3.7+
- [Ollama](https://ollama.ai/) installed and running
- A model pulled into Ollama (default: llama3)

## Setup

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the application**: [http://localhost:5000](http://localhost:5000)

## Usage

1. Ensure Ollama is running
2. Enter a YouTube URL
3. Process the video to extract the transcript
4. Ask questions or generate a summary

## Configuration

Create a `.env` file with:
```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT=30
DEBUG=False
```

## How It Works

1. **Transcript Extraction**: Uses YouTube API or Whisper for transcription
2. **RAG Processing**: Chunks transcript, maintains timestamps, stores in ChromaDB
3. **Question Answering**: Finds relevant chunks, retrieves timestamps, generates answers
4. **Summarization**: Selects representative chunks for summarization

## API Endpoints

- `POST /process_video`: Process a YouTube video
- `POST /ask`: Ask a question
- `POST /summarize`: Generate a summary
- `GET /get_available_models`: List available models
- `GET /get_processed_videos`: List processed videos
- `GET /debug`: Connection status
- `GET /config`: Current configuration
