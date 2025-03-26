# YouTube RAG - Enhanced with Knowledge Graphs

A Retrieval-Augmented Generation (RAG) application for YouTube videos with LangGraph-powered knowledge graph integration.

## Features

- Process YouTube videos to extract transcripts (using YouTube API or Whisper speech-to-text)
- Create semantic search indexes for video content
- Ask questions about video content with precise timestamp references
- Generate summaries of videos
- **NEW:** Build knowledge graphs from video content for enhanced understanding
- **NEW:** Use structured knowledge to provide more accurate and contextual answers

## Knowledge Graph Features

The application now includes a LangGraph-powered knowledge graph builder that:

1. Extracts entities (people, concepts, products, organizations) from video transcripts
2. Identifies relationships between these entities
3. Stores this structured information in either:
   - Neo4j graph database (recommended for production)
   - In-memory graph (for simple deployments)
4. Uses this knowledge graph to enhance RAG responses

When enabled, the knowledge graph:
- Provides structured understanding of video content
- Helps identify key concepts and their relationships
- Improves answer accuracy by incorporating structured knowledge
- Enables more contextual responses

## Requirements

- Python 3.8+
- Ollama for local LLM inference
- FFmpeg for audio processing (when using Whisper)
- Optional: Neo4j database (for scalable knowledge graph storage)

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

# Knowledge Graph configuration
USE_KNOWLEDGE_GRAPH=True
# For Neo4j (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

## API Endpoints

### Core Endpoints

- `POST /process_video` - Process a YouTube video
- `POST /ask` - Ask a question about a video
- `POST /summarize` - Generate a summary of a video
- `GET /get_processed_videos` - List all processed videos

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
