# YouTube RAG

A Retrieval-Augmented Generation (RAG) application for YouTube videos.

## Features

- Process YouTube videos to extract transcripts (using YouTube API or Whisper speech-to-text)
- Create semantic search indexes for video content
- Ask questions about video content with precise timestamp references
- Generate summaries of videos
- Click on timestamps in answers to jump to specific parts of videos

## Requirements

- Python 3.8+
- Ollama for local LLM inference
- FFmpeg for audio processing (when using Whisper)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start Ollama on your system (see https://ollama.ai)
4. Pull a model in Ollama:
   ```
   ollama pull llama3
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
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=30
```

## API Endpoints

### Core Endpoints

- `POST /process_video` - Process a YouTube video
- `POST /ask` - Ask a question about a video
- `POST /summarize` - Generate a summary of a video
- `GET /get_processed_videos` - List all processed videos
<<<<<<< Updated upstream

### Knowledge Graph Endpoints

- `GET /kg_info` - Get knowledge graph statistics
- `GET /kg_entities` - Get entities and relationships for a video

## How Knowledge Graphs Enhance RAG

1. **Traditional RAG** uses embeddings to find similar chunks of text, but often struggles with:
   - Understanding relationships between concepts
   - Following complex reasoning chains
   - Providing structured answers

2. **Knowledge Graph Enhanced RAG** adds structured understanding by:
   - Identifying key entities in the content
   - Mapping relationships between these entities
   - Using this structured data to enhance responses

For example, when asked about a complex topic from a video, the system can:
- Retrieve relevant text chunks (traditional RAG)
- Enhance the answer with relevant entity relationships from the knowledge graph
- Provide more accurate and structured responses
=======
- `GET /get_chunks` - Get transcript chunks for a video

## License

MIT
>>>>>>> Stashed changes
