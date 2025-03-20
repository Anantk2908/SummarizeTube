# YouTube RAG with Ollama

This application allows you to process YouTube videos, extract their transcripts, and perform question-answering and summarization using Ollama LLMs.

## Features

- Extract transcripts with timestamps from YouTube videos using either:
  - YouTube's transcript API (when available)
  - Whisper for speech-to-text transcription (when API transcripts unavailable)
- Process and chunk transcripts for efficient retrieval
- Ask questions about the video content using RAG
- Get answers with relevant timestamps and direct links to those points in the video
- Generate summaries of the video content
- Clean, responsive web interface
- Support for multiple Ollama models
- Environment variable configuration via .env file

## Prerequisites

- Python 3.7 or higher
- [Ollama](https://ollama.ai/) installed and running
- A model pulled into Ollama (default: llama3)

## Getting Started

There are scripts to help you set up and run the application automatically. Choose the appropriate script for your operating system:

### On Linux/macOS:

```bash
./run_youtube_rag.sh
```

### On Windows (PowerShell):

```powershell
.\run_youtube_rag.ps1
```

These scripts will:
1. Create a Python virtual environment (if needed)
2. Install the required dependencies
3. Start the Flask server
4. Open your default browser to the application

## Manual Setup

If you prefer to run things manually:

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```

3. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the application**:
   Open your browser and navigate to [http://localhost:5000](http://localhost:5000)

## Usage

1. Ensure Ollama is running
2. Enter a YouTube URL in the input field
3. Click "Process Video" to extract and store the transcript
4. Ask questions about the video or generate a summary

## Configuration

You can configure the application by creating a `.env` file with the following options:

```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=30
DEBUG=False
```

## Troubleshooting

- If you see the error "Ollama not available", make sure Ollama is running
- Click the "Debug" button to see detailed connection information
- Check that you have pulled the model you're trying to use (e.g., `ollama pull llama3`)

## Cross-platform Usage

### Using Ollama on Windows from WSL

If you're running the application in WSL but Ollama is installed on Windows:

1. Find your Windows host IP (typically something like 192.168.x.x)
2. Set the OLLAMA_HOST in .env to http://YOUR_WINDOWS_IP:11434
3. Make sure Windows Firewall allows connections to port 11434

## How It Works

1. **Transcript Extraction**:
   - First attempts to use the YouTube Transcript API to get captions
   - If that fails, downloads the audio and uses Whisper for transcription

2. **RAG Processing**:
   - Splits the transcript into manageable chunks
   - Maintains timestamp information for each chunk
   - Stores chunks with metadata in ChromaDB with semantic embeddings

3. **Question Answering**:
   - Uses semantic search to find relevant chunks for a query
   - Retrieves associated timestamps
   - Sends relevant context to Ollama for answer generation
   - Returns the answer with clickable timestamps

4. **Summarization**:
   - Selects representative chunks from the full transcript
   - Sends them to Ollama with a summarization prompt
   - Returns a concise summary of the video content

## Debugging

If you encounter connection issues with Ollama:

1. Visit the `/debug` endpoint to see detailed connection information
2. Check your Ollama host configuration in the .env file
3. Ensure Ollama is running and accessible from your machine

## API Endpoints

The application exposes the following API endpoints:

- `POST /process_video`: Process a YouTube video (extract transcript, create embeddings)
- `POST /ask`: Ask a question about a processed video
- `POST /summarize`: Generate a summary of a processed video
- `GET /get_available_models`: Get a list of available Ollama models
- `GET /get_processed_videos`: Get a list of previously processed videos
- `GET /debug`: Get detailed information about Ollama connection status
- `GET /config`: Get current application configuration

## Limitations

- Requires videos to have transcripts or clear audio for transcription
- Performance depends on the Ollama model being used
- Transcription quality may vary for non-English content
- Whisper transcription can be slow on systems without GPU

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running and your OLLAMA_HOST is correctly set
- **Model Not Found**: Use `ollama pull <model-name>` to download missing models
- **Transcription Errors**: Check that FFmpeg is installed and in your PATH
- **Memory Issues**: Consider using a smaller Whisper model or a lighter Ollama model 