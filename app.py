import os
import yt_dlp
import whisper
import chromadb
import json
import time
import requests
import socket
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
import traceback
# Import the knowledge graph builder
from kg_builder import KnowledgeGraphBuilder

# Load environment variables from .env file
load_dotenv()

# Configure Ollama from environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
DEBUG_MODE = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
# Streaming has been disabled to avoid connection issues
ENABLE_STREAMING = False  # Manually disabled to avoid connection issues

# Neo4j configuration (optional)
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
USE_KNOWLEDGE_GRAPH = os.getenv("USE_KNOWLEDGE_GRAPH", "True").lower() in ("true", "1", "t")

# Set the Ollama host environment variable for the ollama client package
if "://" in OLLAMA_HOST:
    # If OLLAMA_HOST contains a protocol, use it as is
    os.environ["OLLAMA_HOST"] = OLLAMA_HOST
else:
    # If it's just an IP or hostname, add the protocol
    os.environ["OLLAMA_HOST"] = f"http://{OLLAMA_HOST}"

app = Flask(__name__)
app.static_folder = 'static'

# Initialize ChromaDB with sentence-transformers embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chromadb_store")
collection = chroma_client.get_or_create_collection(
    name="youtube_transcripts",
    embedding_function=sentence_transformer_ef
)

# Initialize Whisper model
whisper_model = None  # Lazy loading to save memory

# Initialize knowledge graph builder if enabled
kg_builder = None
if USE_KNOWLEDGE_GRAPH:
    try:
        kg_builder = KnowledgeGraphBuilder(
            model_name=OLLAMA_MODEL,
            neo4j_uri=NEO4J_URI if NEO4J_URI else None,
            neo4j_username=NEO4J_USERNAME if NEO4J_USERNAME else None,
            neo4j_password=NEO4J_PASSWORD if NEO4J_PASSWORD else None
        )
        print("Knowledge Graph builder initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize Knowledge Graph builder: {e}")
        print("Will run without knowledge graph capabilities")
        USE_KNOWLEDGE_GRAPH = False

# Function to extract video ID from URL
def extract_video_id(youtube_url):
    """
    Extract video ID from various YouTube URL formats:
    - youtube.com/watch?v=...
    - youtu.be/...
    - youtube.com/shorts/...
    - youtube.com/embed/...
    - youtube.com/v/...
    """
    if not youtube_url:
        return None
        
    # Remove any whitespace and get the base URL
    youtube_url = youtube_url.strip()
    
    # Handle URLs with or without protocol
    if youtube_url.startswith('//'):
        youtube_url = 'https:' + youtube_url
    elif not youtube_url.startswith(('http://', 'https://')):
        youtube_url = 'https://' + youtube_url
    
    try:
        # Parse the URL
        parsed_url = urlparse(youtube_url)
        
        # Handle youtu.be URLs
        if parsed_url.netloc == 'youtu.be':
            return parsed_url.path.strip('/')
            
        # Handle various youtube.com formats
        if parsed_url.netloc in ['youtube.com', 'www.youtube.com']:
            # Handle /watch URLs
            if parsed_url.path == '/watch':
                query = parse_qs(parsed_url.query)
                return query.get('v', [None])[0]
                
            # Handle /shorts/, /embed/, and /v/ URLs
            for path_prefix in ['/shorts/', '/embed/', '/v/']:
                if parsed_url.path.startswith(path_prefix):
                    return parsed_url.path.replace(path_prefix, '').split('/')[0]
        
        return None
        
    except Exception as e:
        print(f"Error extracting video ID: {str(e)}")
        return None

# Function to get transcript with timestamps (if available)
def get_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Create full transcript text
        full_text = " ".join([t["text"] for t in transcript_list])
        
        # Store timestamps info
        timestamps = []
        for entry in transcript_list:
            timestamps.append({
                "text": entry["text"],
                "start": entry["start"],
                "duration": entry["duration"],
                "formatted_time": format_time(entry["start"])
            })
        
        return full_text, timestamps
    except Exception as e:
        print(f"Error fetching transcript via API: {str(e)}")
        return None, None

# Debug function to test connectivity
def test_ollama_connection():
    host = OLLAMA_HOST.replace("http://", "").replace("https://", "").split(":")[0]
    port = 11434  # Default Ollama port
    
    try:
        # Test basic socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ Port {port} is open on {host}")
            # Try HTTP request
            try:
                response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
                print(f"✅ HTTP request successful! Status code: {response.status_code}")
                return {
                    "socket_test": "Success",
                    "http_test": "Success",
                    "status_code": response.status_code,
                    "response": response.json() if response.status_code == 200 else None
                }
            except requests.exceptions.RequestException as e:
                print(f"❌ HTTP request failed: {str(e)}")
                return {
                    "socket_test": "Success",
                    "http_test": "Failed",
                    "error": str(e)
                }
        else:
            print(f"❌ Port {port} is closed on {host}")
            return {
                "socket_test": "Failed",
                "error": f"Port {port} is closed on {host}"
            }
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")
        return {
            "socket_test": "Error",
            "error": str(e)
        }

# Function to format seconds into HH:MM:SS
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Function to download and transcribe audio
def transcribe_video(video_id):
    global whisper_model
    
    # Load whisper model if not already loaded
    if whisper_model is None:
        try:
            whisper_model = whisper.load_model("small")
        except Exception as e:
            return None, None, f"Error loading Whisper model: {str(e)}"
    
    try:
        output_path = f"{video_id}.mp3"
        
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
            "outtmpl": output_path
        }

        # Download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            title = info.get('title', f'Video {video_id}')

        # Transcribe with whisper
        result = whisper_model.transcribe(output_path)
        
        # Generate timestamps (whisper provides segmentation)
        timestamps = []
        if 'segments' in result:
            for segment in result['segments']:
                timestamps.append({
                    "text": segment['text'],
                    "start": segment['start'],
                    "duration": segment['end'] - segment['start'],
                    "formatted_time": format_time(segment['start'])
                })
        
        # Clean up downloaded file
        if os.path.exists(output_path):
            os.remove(output_path)
            
        return result["text"], timestamps, title
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(output_path):
            os.remove(output_path)
        return None, None, f"Error transcribing: {str(e)}"

# Function to chunk transcript and add to database
def process_transcript_for_rag(video_id, transcript, timestamps, title, youtube_url):
    try:
        # Delete existing entries for this video
        try:
            collection.delete(ids=[f"{video_id}_chunk_{i}" for i in range(1000)])  # Attempt to delete potential old chunks
        except:
            pass  # Ignore if no entries exist
        
        # Split transcript into chunks for better RAG
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(transcript)
        
        # Prepare metadata and chunks for storage
        ids = []
        metadatas = []
        documents = []
        
        # Assign timestamps to chunks
        total_len = len(transcript)
        curr_pos = 0
        chunk_positions = []
        
        # Calculate starting positions of each chunk in the full text
        for chunk in chunks:
            chunk_positions.append(curr_pos / total_len)  # Store as percentage position
            curr_pos += len(chunk)
        
        # Store chunks with metadata
        for i, chunk in enumerate(chunks):
            chunk_id = f"{video_id}_chunk_{i}"
            position = chunk_positions[i]
            
            # Find closest timestamp for this chunk
            matching_timestamp = None
            for ts in timestamps:
                # Calculate approximate position of this timestamp in the transcript
                ts_pos = ts["start"] / timestamps[-1]["start"] if timestamps[-1]["start"] > 0 else 0
                
                # Use the first timestamp or update if this is closer
                if matching_timestamp is None or abs(ts_pos - position) < abs(matching_timestamp["start"] / timestamps[-1]["start"] - position):
                    matching_timestamp = ts
            
            # Add to collections
            ids.append(chunk_id)
            metadatas.append({
                "video_id": video_id,
                "url": youtube_url,
                "title": title,
                "chunk_id": i,
                "timestamp": matching_timestamp["start"] if matching_timestamp else 0,
                "formatted_time": matching_timestamp["formatted_time"] if matching_timestamp else "00:00:00",
                "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
            documents.append(chunk)
        
        # Add to database
        collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=documents
        )
        
        # Process for knowledge graph if enabled
        if USE_KNOWLEDGE_GRAPH and kg_builder:
            try:
                print(f"Processing transcript for knowledge graph: {len(transcript)} chars")
                kg_result = kg_builder.process_text(transcript)
                print(f"Extracted {len(kg_result.get('entities', []))} entities and {len(kg_result.get('relationships', []))} relationships")
            except Exception as e:
                print(f"Error processing for knowledge graph: {e}")
        
        return True, len(chunks)
    except Exception as e:
        print(f"Error processing transcript: {str(e)}")
        return False, str(e)

# Check if Ollama is running
def check_ollama_available():
    try:
        if DEBUG_MODE:
            print(f"Attempting to connect to Ollama at {OLLAMA_HOST}/api/tags")
        
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=OLLAMA_TIMEOUT)
        
        if DEBUG_MODE:
            print(f"Ollama connection response: {response.status_code}")
        
        return response.status_code == 200
    except Exception as e:
        if DEBUG_MODE:
            print(f"Error connecting to Ollama: {str(e)}")
            # Run detailed connection test
            test_results = test_ollama_connection()
            print(f"Connection test results: {json.dumps(test_results, indent=2)}")
        
        return False

# Function to check if a specific model is available in Ollama
def check_model_available(model_name):
    try:
        if DEBUG_MODE:
            print(f"Checking if model {model_name} is available")
        
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=OLLAMA_TIMEOUT)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available = any(model["name"] == model_name for model in models)
            
            if DEBUG_MODE:
                print(f"Available models: {[model['name'] for model in models]}")
                print(f"Model {model_name} available: {available}")
            
            return available
        return False
    except Exception as e:
        if DEBUG_MODE:
            print(f"Error checking model availability: {str(e)}")
        return False

# Define routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/debug", methods=["GET"])
def debug():
    """Debug endpoint to check Ollama connectivity"""
    connection_test = test_ollama_connection()
    ollama_status = check_ollama_available()
    
    if ollama_status:
        try:
            models_response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=OLLAMA_TIMEOUT)
            models = models_response.json().get("models", [])
            model_names = [model["name"] for model in models]
        except:
            model_names = []
    else:
        model_names = []
    
    return jsonify({
        "ollama_status": ollama_status,
        "ollama_host": OLLAMA_HOST,
        "ollama_url": OLLAMA_HOST,
        "connection_test": connection_test,
        "available_models": model_names,
        "default_model": OLLAMA_MODEL,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/process_video", methods=["POST"])
def process_video():
    data = request.json
    youtube_url = data.get("url")
    if not youtube_url:
        return jsonify({"error": "No URL provided"}), 400

    video_id = extract_video_id(youtube_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400
    
    # Check for Ollama
    if not check_ollama_available():
        return jsonify({"error": f"Ollama is not running at {OLLAMA_HOST}. Please start Ollama server."}), 500

    # Check if model is available
    model_name = data.get("model", OLLAMA_MODEL)
    if not check_model_available(model_name):
        return jsonify({
            "error": f"Model '{model_name}' not found in Ollama. Please run 'ollama pull {model_name}' first."
        }), 400
    
    # Get video info
    video_title = f"Video {video_id}"
    try:
        with yt_dlp.YoutubeDL({"skip_download": True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            video_title = info.get('title', video_title)
    except:
        pass  # Ignore errors, we'll just use the default title
    
    # First try to get transcript via YouTube API
    transcript, timestamps = get_youtube_transcript(video_id)
    source = "youtube_api"

    # If no transcript, use Whisper
    if not transcript:
        transcript, timestamps, whisper_result = transcribe_video(video_id)
        source = "whisper"
        if isinstance(whisper_result, str) and whisper_result.startswith("Error"):
            return jsonify({"error": whisper_result}), 500
        if whisper_result and not video_title.startswith("Video "):
            video_title = whisper_result
    
    if not transcript or not timestamps:
        return jsonify({"error": "Failed to obtain transcript"}), 500
    
    # Process transcript for RAG
    success, result = process_transcript_for_rag(video_id, transcript, timestamps, video_title, youtube_url)
    if not success:
        return jsonify({"error": f"Failed to process transcript: {result}"}), 500

    return jsonify({
        "message": "Processed successfully", 
        "video_id": video_id,
        "title": video_title,
        "chunks": result,
        "source": source
    })

# Define prompt templates
QA_PROMPT_TEMPLATE = """You are a highly knowledgeable AI assistant that provides accurate, detailed answers about YouTube video content.

Your task is to answer questions about the video using ONLY the provided transcript segments. Follow these guidelines:

1. Use ONLY the information from the provided transcript segments
2. If the exact information isn't in the segments, say "This information is not in the video segments provided."
3. If you can make a reasonable inference from the segments, start with "Based on the context..."
4. Look for related terminology or descriptions that may answer the question indirectly
5. If the video compares different models, consider if the answer might be implied when discussing features
6. Keep answers concise but complete
7. If segments contain technical terms or specific names, maintain their exact usage
8. If segments show chronological events, maintain the correct sequence
9. If the answer requires combining information from multiple segments, ensure logical connection
10. For questions about motorcycle features, check for discussions about specifications, comparisons, or riding experience

TRANSCRIPT SEGMENTS:
{context}

QUESTION: {question}

ANSWER:"""

SUMMARY_PROMPT_TEMPLATE = """You are an expert content analyst creating high-quality video summaries.

Create a comprehensive summary of the YouTube video titled "{title}". Follow these guidelines:

1. Structure the summary in 1-2 clear paragraphs
2. Begin with the main topic or purpose of the video
3. Highlight key points in order of importance
4. Include specific details, numbers, or examples when available
5. Maintain any technical terminology used in the video
6. If the content is instructional, outline the main steps or concepts
7. If it's a discussion, capture the main arguments or viewpoints
8. End with the video's conclusion or key takeaway
9. Use clear, professional language
10. Do NOT mention that this is based on transcript excerpts

TRANSCRIPT EXCERPTS:
{context}

SUMMARY:"""

# Function to stream response from Ollama API directly
def stream_ollama_response(model_name, prompt):
    # Create a direct request to Ollama API for streaming
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    # Get host from environment variable without the protocol
    host = OLLAMA_HOST
    
    try:
        with requests.post(f"{host}/api/chat", json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as response:
            if response.status_code != 200:
                yield f"data: {json.dumps({'error': 'Error connecting to Ollama API'})}\n\n"
                return
                
            # Initial variables
            full_response = ""
            
            # Stream each chunk as it arrives
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    json_chunk = json.loads(line)
                    
                    # Extract the delta text from the chunk
                    if 'message' in json_chunk and 'content' in json_chunk['message']:
                        chunk_text = json_chunk['message']['content']
                        full_response += chunk_text
                        
                        yield f"data: {json.dumps({'content': chunk_text, 'done': False})}\n\n"
                        
                    # Check if we're done
                    if json_chunk.get('done', False):
                        yield f"data: {json.dumps({'content': '', 'done': True, 'full_response': full_response})}\n\n"
                        break
                        
                except json.JSONDecodeError:
                    # Skip any malformed lines
                    continue
    except Exception as e:
        if DEBUG_MODE:
            print(f"Error in streaming: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.route("/ask_stream", methods=["POST"])
def ask_question_stream():
    """Streaming version of ask_question endpoint"""
    if not ENABLE_STREAMING:
        return jsonify({"error": "Streaming is disabled"}), 400
        
    data = request.json
    video_id = data.get("video_id")
    question = data.get("question")
    model_name = data.get("model", OLLAMA_MODEL)

    if not video_id or not question:
        return jsonify({"error": "Missing parameters"}), 400
    
    # Check for Ollama
    if not check_ollama_available():
        return jsonify({"error": f"Ollama is not running at {OLLAMA_HOST}. Please start Ollama server."}), 500
    
    # Check if model is available
    if not check_model_available(model_name):
        return jsonify({
            "error": f"Model '{model_name}' not found in Ollama. Please run 'ollama pull {model_name}' first."
        }), 400

    # Query database for relevant chunks
    results = collection.query(
        query_texts=[question], 
        where={"video_id": video_id},
        n_results=5,
        include=["metadatas", "documents"]
    )

    if not results["documents"] or len(results["documents"][0]) == 0:
        return jsonify({"error": "No content found for this video ID"}), 404

    # Prepare context from retrieved chunks
    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)
    
    # Get timestamps
    timestamps = []
    for metadata in results["metadatas"][0]:
        timestamps.append({
            "time": metadata["timestamp"],
            "formatted_time": metadata["formatted_time"],
            "text_preview": metadata["text_preview"]
        })
    
    # Format prompt using the template
    prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)

    # Set response headers for SSE
    def generate():
        # First yield the timestamps so frontend has them right away
        yield f"data: {json.dumps({'timestamps': timestamps, 'type': 'timestamps'})}\n\n"
        
        # Then yield the content as it streams
        yield from stream_ollama_response(model_name, prompt)
    
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache', 
                            'X-Accel-Buffering': 'no'})

@app.route("/summarize_stream", methods=["POST"])
def summarize_video_stream():
    """Streaming version of summarize endpoint"""
    if not ENABLE_STREAMING:
        return jsonify({"error": "Streaming is disabled"}), 400
        
    data = request.json
    video_id = data.get("video_id")
    model_name = data.get("model", OLLAMA_MODEL)

    if not video_id:
        return jsonify({"error": "Missing video_id"}), 400
    
    # Check for Ollama
    if not check_ollama_available():
        return jsonify({"error": f"Ollama is not running at {OLLAMA_HOST}. Please start Ollama server."}), 500
    
    # Check if model is available
    if not check_model_available(model_name):
        return jsonify({
            "error": f"Model '{model_name}' not found in Ollama. Please run 'ollama pull {model_name}' first."
        }), 400

    # First check if we already have a cached summary
    try:
        cached_results = collection.get(
            where={"video_id": video_id, "type": "summary"},
            include=["documents", "metadatas"]
        )
        
        if cached_results["documents"] and len(cached_results["documents"]) > 0:
            # Return cached summary as a stream event
            print(f"Returning cached summary for video {video_id}")
            
            # Get video title from metadata
            video_title = cached_results["metadatas"][0]["title"] if cached_results["metadatas"] else f"Video {video_id}"
            summary_text = cached_results["documents"][0]
            created_at = cached_results["metadatas"][0].get("created_at", "unknown time")
            
            # Set response headers for SSE
            def generate_cached():
                # First yield the title
                yield f"data: {json.dumps({'title': video_title, 'type': 'title'})}\n\n"
                
                # Then yield the cached content
                yield f"data: {json.dumps({'content': summary_text, 'cached': True, 'created_at': created_at})}\n\n"
                
                # Signal completion
                yield f"data: {json.dumps({'content': '', 'done': True, 'full_response': summary_text})}\n\n"
            
            return Response(stream_with_context(generate_cached()), 
                           mimetype='text/event-stream',
                           headers={'Cache-Control': 'no-cache', 
                                   'X-Accel-Buffering': 'no'})
    except Exception as e:
        print(f"Error checking for cached summary: {str(e)}")
        # Continue with generating a new summary

    # Query database for all chunks
    results = collection.query(
        query_texts=["What is this video about?"],  # General query to get relevant chunks
        where={"video_id": video_id},
        n_results=10,
        include=["metadatas", "documents"]
    )

    if not results["documents"] or len(results["documents"][0]) == 0:
        return jsonify({"error": "No content found for this video ID"}), 404

    # Get video title
    video_title = results["metadatas"][0][0]["title"] if results["metadatas"] and results["metadatas"][0] else f"Video {video_id}"
    
    # Get representative chunks for summary (first, middle, and last parts)
    chunks = results["documents"][0]
    
    # If too many chunks, select representative ones
    selected_chunks = chunks
    if len(chunks) > 5:
        # Take first, some from middle, and last chunk
        selected_chunks = [
            chunks[0],
            chunks[len(chunks)//4],
            chunks[len(chunks)//2],
            chunks[3*len(chunks)//4],
            chunks[-1]
        ]
    
    context = "\n\n".join(selected_chunks)
    
    # Format summary prompt using the template
    prompt = SUMMARY_PROMPT_TEMPLATE.format(title=video_title, context=context)

    # Set response headers for SSE
    def generate():
        # First yield the title so frontend has it right away
        yield f"data: {json.dumps({'title': video_title, 'type': 'title'})}\n\n"
        
        # Store for saving the complete summary
        full_summary = ""
        
        # Stream each chunk as it arrives
        for chunk in stream_ollama_response(model_name, prompt):
            yield chunk
            
            # Extract content from chunk to build full summary
            try:
                chunk_data = json.loads(chunk.replace('data: ', ''))
                if 'content' in chunk_data and chunk_data['content']:
                    full_summary += chunk_data['content']
                
                # If this is the final chunk, save the summary
                if chunk_data.get('done', False) and full_summary:
                    try:
                        timestamp = datetime.now().isoformat()
                        collection.upsert(
                            ids=[f"{video_id}_summary"],
                            metadatas=[{
                                "video_id": video_id,
                                "title": video_title,
                                "type": "summary",
                                "created_at": timestamp
                            }],
                            documents=[full_summary]
                        )
                        print(f"Cached summary for video {video_id}")
                    except Exception as e:
                        print(f"Error caching summary: {str(e)}")
            except:
                pass  # Skip any parsing errors
    
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache', 
                            'X-Accel-Buffering': 'no'})

@app.route("/get_available_models", methods=["GET"])
def get_available_models():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=OLLAMA_TIMEOUT)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            return jsonify({"models": model_names, "default_model": OLLAMA_MODEL})
        return jsonify({"error": "Failed to get models", "models": [OLLAMA_MODEL], "default_model": OLLAMA_MODEL}), 200
    except Exception as e:
        print(f"Error getting models: {str(e)}")
        return jsonify({"error": f"Ollama not available at {OLLAMA_HOST}", "models": [OLLAMA_MODEL], "default_model": OLLAMA_MODEL}), 200

@app.route("/get_processed_videos", methods=["GET"])
def get_processed_videos():
    try:
        # Query for unique video IDs
        all_metadatas = collection.get()["metadatas"]
        processed_videos = {}
        
        for metadata in all_metadatas:
            if "video_id" in metadata and "title" in metadata and "url" in metadata:
                video_id = metadata["video_id"]
                if video_id not in processed_videos:
                    processed_videos[video_id] = {
                        "id": video_id,
                        "title": metadata["title"],
                        "url": metadata["url"]
                    }
        
        return jsonify({"videos": list(processed_videos.values())})
    except Exception as e:
        return jsonify({"error": f"Error fetching videos: {str(e)}"}), 500

@app.route("/config", methods=["GET"])
def get_config():
    """Endpoint to retrieve current configuration (except sensitive data)"""
    return jsonify({
        "ollama_host": OLLAMA_HOST,
        "default_model": OLLAMA_MODEL,
        "debug_mode": DEBUG_MODE,
        "timeout": OLLAMA_TIMEOUT
    })

@app.route("/get_chunks", methods=["GET"])
def get_chunks():
    """Endpoint to retrieve all chunks for a specific video"""
    video_id = request.args.get("video_id")
    
    if not video_id:
        return jsonify({"error": "Missing video_id parameter"}), 400
    
    try:
        # Get all chunks for this video
        all_chunks = collection.get(
            where={"video_id": video_id},
            include=["metadatas", "documents"]
        )
        
        if not all_chunks["documents"]:
            return jsonify({"error": "No chunks found for this video ID"}), 404
        
        # Prepare response with chunks and their metadata, filtering out summaries
        chunks = []
        for i, doc in enumerate(all_chunks["documents"]):
            if i < len(all_chunks["metadatas"]):
                metadata = all_chunks["metadatas"][i]
                # Only include if it's not a summary document
                if metadata.get("type") != "summary":
                    chunks.append({
                        "id": len(chunks),  # Use new index for filtered list
                        "text": doc,
                        "timestamp": metadata.get("timestamp", 0),
                        "formatted_time": metadata.get("formatted_time", "00:00:00"),
                        "metadata": metadata
                    })
        
        if not chunks:
            return jsonify({"error": "No transcript chunks found for this video"}), 404
            
        # Sort chunks by timestamp
        chunks.sort(key=lambda x: x["timestamp"])
        
        return jsonify({
            "video_id": video_id,
            "chunks": chunks,
            "count": len(chunks)
        })
        
    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error retrieving chunks: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    video_id = data.get("video_id")
    question = data.get("question")
    model_name = data.get("model", OLLAMA_MODEL)
    use_kg = data.get("use_kg", USE_KNOWLEDGE_GRAPH)  # Allow override via API

    if not video_id or not question:
        return jsonify({"error": "Missing parameters"}), 400
    
    # Check for Ollama
    if not check_ollama_available():
        return jsonify({"error": f"Ollama is not running at {OLLAMA_HOST}. Please start Ollama server."}), 500
    
    # Check if model is available
    if not check_model_available(model_name):
        return jsonify({
            "error": f"Model '{model_name}' not found in Ollama. Please run 'ollama pull {model_name}' first."
        }), 400

    try:
        print(f"Querying ChromaDB for video_id: {video_id}")
        
        # First get all chunks for this video (without $not operator which isn't supported in get())
        all_chunks = collection.get(
            where={"video_id": video_id},
            include=["metadatas", "documents"]
        )
        
        if not all_chunks["documents"]:
            return jsonify({"error": "No content found for this video ID"}), 404
            
        # Filter out summaries manually (since we can't use $not in get())
        filtered_docs = []
        filtered_metadatas = []
        
        for i, metadata in enumerate(all_chunks["metadatas"]):
            if metadata.get("type") != "summary":
                filtered_docs.append(all_chunks["documents"][i])
                filtered_metadatas.append(metadata)
        
        if not filtered_docs:
            return jsonify({"error": "No transcript chunks found for this video"}), 404
            
        print(f"Found {len(filtered_docs)} transcript chunks for video")
        
        # Then query for relevant chunks
        results = collection.query(
            query_texts=[question],
            where={"video_id": video_id},
            n_results=12,  # Increased from 8 to get more candidates
            include=["metadatas", "documents", "distances"]
        )
        
        print(f"Query returned {len(results['documents'][0])} chunks")
        
        # Filter out summaries from query results
        filtered_results_docs = []
        filtered_results_metadatas = []
        filtered_results_distances = []
        
        for i, metadata in enumerate(results["metadatas"][0]):
            if metadata.get("type") != "summary":
                filtered_results_docs.append(results["documents"][0][i])
                filtered_results_metadatas.append(metadata)
                if "distances" in results and results["distances"]:
                    filtered_results_distances.append(results["distances"][0][i])
        
        if not filtered_results_docs:
            return jsonify({"error": "No relevant transcript chunks found for this question"}), 404
            
        print(f"After filtering, using {len(filtered_results_docs)} relevant chunks")

        # Get chunks and their timestamps
        chunks_with_time = []
        for i, doc in enumerate(filtered_results_docs):
            metadata = filtered_results_metadatas[i]
            score = float(filtered_results_distances[i]) if filtered_results_distances else 1.0
            chunks_with_time.append({
                "text": doc,
                "timestamp": float(metadata.get("timestamp", 0)),
                "formatted_time": metadata.get("formatted_time", "00:00:00"),
                "text_preview": metadata.get("text_preview", ""),
                "score": score
            })
            print(f"Chunk {i}: score={score:.4f}, time={metadata.get('formatted_time', '00:00:00')}, preview={metadata.get('text_preview', '')[:50]}")

        # Sort chunks by timestamp to maintain chronological order
        chunks_with_time.sort(key=lambda x: x["timestamp"])

        # Select chunks based on relevance and continuity
        selected_chunks = []
        timestamps = []
        
        # Use a more forgiving relevance threshold to include more potentially relevant chunks
        relevance_threshold = 0.5  # Increased from 0.3
        
        # First add highly relevant chunks (low distance score)
        for chunk in chunks_with_time:
            if chunk["score"] < relevance_threshold:
                selected_chunks.append(chunk["text"])
                timestamps.append({
                    "time": chunk["timestamp"],
                    "formatted_time": chunk["formatted_time"],
                    "text_preview": chunk["text_preview"]
                })
                print(f"Selected chunk with score {chunk['score']:.4f}: {chunk['text_preview'][:50]}")
        
        # If we don't have enough chunks by relevance, add the most relevant ones regardless of threshold
        if len(selected_chunks) < 3:
            # Sort by relevance
            sorted_by_relevance = sorted(chunks_with_time, key=lambda x: x["score"])
            for chunk in sorted_by_relevance:
                if chunk["text"] not in selected_chunks and len(selected_chunks) < 5:
                    selected_chunks.append(chunk["text"])
                    timestamps.append({
                        "time": chunk["timestamp"],
                        "formatted_time": chunk["formatted_time"],
                        "text_preview": chunk["text_preview"]
                    })
                    print(f"Added chunk by relevance sorting: {chunk['text_preview'][:50]}")

        if not selected_chunks:
            return jsonify({"error": "Could not find relevant content for this question"}), 404
            
        print(f"Selected {len(selected_chunks)} chunks for context")

        # Combine chunks with paragraph breaks for better readability
        context = "\n\n".join(selected_chunks)
        
        # Generate answer with knowledge graph if enabled, or use regular prompt if not
        start_time = time.time()
        answer_text = ""
        
        if use_kg and USE_KNOWLEDGE_GRAPH and kg_builder:
            # Use knowledge graph enhanced answer
            print("Using knowledge graph enhanced answer")
            answer_text = kg_builder.get_enhanced_answer(question, context, video_id)
        else:
            # Format regular prompt using the template
            prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)

            print(f"Sending prompt to Ollama: {prompt[:200]}...")  # Print first 200 chars of prompt

            # Query Ollama with error handling
            try:
                response = ollama.chat(
                    model=model_name, 
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract relevant data from ChatResponse object
                response_data = {
                    "model": getattr(response, "model", model_name),
                    "created_at": getattr(response, "created_at", None),
                    "eval_duration": getattr(response, "eval_duration", None),
                    "message": {
                        "role": response.message.role if hasattr(response.message, "role") else "assistant",
                        "content": response.message.content if hasattr(response.message, "content") else None
                    }
                }
                
                print(f"Ollama response data: {json.dumps(response_data, indent=2)}")
                
                # Validate response content
                if not response.message or not hasattr(response.message, "content"):
                    raise ValueError("Response missing content")
                    
                answer_text = response.message.content
                if not answer_text:
                    raise ValueError("Empty answer content")
            except Exception as e:
                print(f"Error in Ollama chat: {str(e)}")
                if 'response' in locals():
                    print(f"Response content: {getattr(response.message, 'content', 'No content available')}")
                raise ValueError(f"Failed to get valid response from Ollama: {str(e)}")
        
        end_time = time.time()
        
        # Return answer with timestamps for UI
        return jsonify({
            "answer": answer_text,
            "timestamps": timestamps,
            "time_taken": round(end_time - start_time, 2),
            "using_kg": use_kg and USE_KNOWLEDGE_GRAPH and kg_builder is not None
        })
            
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        print(f"Full error details: {traceback.format_exc()}")
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500

@app.route("/summarize", methods=["POST"])
def summarize_video():
    data = request.json
    video_id = data.get("video_id")
    model_name = data.get("model", OLLAMA_MODEL)

    if not video_id:
        return jsonify({"error": "Missing video_id"}), 400
    
    # Check for Ollama
    if not check_ollama_available():
        return jsonify({"error": f"Ollama is not running at {OLLAMA_HOST}. Please start Ollama server."}), 500
    
    # Check if model is available
    if not check_model_available(model_name):
        return jsonify({
            "error": f"Model '{model_name}' not found in Ollama. Please run 'ollama pull {model_name}' first."
        }), 400
    
    # First check if we already have a cached summary
    try:
        cached_results = collection.get(
            where={"video_id": video_id, "type": "summary"},
            include=["documents", "metadatas"]
        )
        
        if cached_results["documents"] and len(cached_results["documents"]) > 0:
            # Return cached summary
            print(f"Returning cached summary for video {video_id}")
            
            # Get video title from metadata
            video_title = cached_results["metadatas"][0]["title"] if cached_results["metadatas"] else f"Video {video_id}"
            
            # Get timestamp when summary was created
            created_at = cached_results["metadatas"][0].get("created_at", "unknown time")
            
            return jsonify({
                "title": video_title,
                "summary": cached_results["documents"][0],
                "cached": True,
                "created_at": created_at
            })
    except Exception as e:
        print(f"Error checking for cached summary: {str(e)}")
        # Continue with generating a new summary

    # Get all chunks for this video
    all_chunks = collection.get(
        where={"video_id": video_id},
        include=["metadatas", "documents"]
    )
    
    if not all_chunks["documents"]:
        return jsonify({"error": "No content found for this video ID"}), 404
        
    # Filter out summaries manually
    transcript_docs = []
    transcript_metadatas = []
    
    for i, metadata in enumerate(all_chunks["metadatas"]):
        if metadata.get("type") != "summary":
            transcript_docs.append(all_chunks["documents"][i])
            transcript_metadatas.append(metadata)
    
    if not transcript_docs:
        return jsonify({"error": "No transcript chunks found for this video"}), 404
        
    # Get video title from any transcript chunk
    video_title = transcript_metadatas[0]["title"] if transcript_metadatas else f"Video {video_id}"
    
    # Query database for relevant chunks for summary
    results = collection.query(
        query_texts=["What is this video about?"],  # General query to get relevant chunks
        where={"video_id": video_id},
        n_results=10,
        include=["metadatas", "documents"]
    )

    if not results["documents"] or len(results["documents"][0]) == 0:
        return jsonify({"error": "No relevant content found for this video ID"}), 404
        
    # Filter out any summaries from the query results
    summary_chunks = []
    
    for i, metadata in enumerate(results["metadatas"][0]):
        if metadata.get("type") != "summary":
            summary_chunks.append(results["documents"][0][i])
    
    if not summary_chunks:
        return jsonify({"error": "No transcript chunks available for summarization"}), 404
    
    # If too many chunks, select representative ones
    selected_chunks = summary_chunks
    if len(summary_chunks) > 5:
        # Take first, some from middle, and last chunk
        selected_chunks = [
            summary_chunks[0],
            summary_chunks[len(summary_chunks)//4],
            summary_chunks[len(summary_chunks)//2],
            summary_chunks[3*len(summary_chunks)//4],
            summary_chunks[-1]
        ]
    
    context = "\n\n".join(selected_chunks)
    
    # Format summary prompt using the template
    prompt = SUMMARY_PROMPT_TEMPLATE.format(title=video_title, context=context)

    try:
        # Generate summary with Ollama
        start_time = time.time()
        response = ollama.chat(
            model=model_name, 
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()
        
        # Get summary text from the response
        if not response.message or not hasattr(response.message, "content"):
            raise ValueError("Response missing content")
            
        summary_text = response.message.content
        if not summary_text:
            raise ValueError("Empty summary content")
        
        # Store the summary in the collection for future use
        try:
            timestamp = datetime.now().isoformat()
            collection.upsert(
                ids=[f"{video_id}_summary"],
                metadatas=[{
                    "video_id": video_id,
                    "title": video_title,
                    "type": "summary",
                    "created_at": timestamp
                }],
                documents=[summary_text]
            )
            print(f"Cached summary for video {video_id}")
        except Exception as e:
            print(f"Error caching summary: {str(e)}")
        
        return jsonify({
            "title": video_title,
            "summary": summary_text,
            "cached": False,
            "time_taken": round(end_time - start_time, 2)
        })
    except Exception as e:
        print(f"Error in summarize_video: {str(e)}")
        print(f"Full error details: {traceback.format_exc()}")
        return jsonify({"error": f"Error generating summary: {str(e)}"}), 500

@app.route("/kg_info", methods=["GET"])
def get_kg_info():
    """Endpoint to get knowledge graph status and statistics"""
    if not USE_KNOWLEDGE_GRAPH or not kg_builder:
        return jsonify({
            "enabled": False,
            "reason": "Knowledge graph is not enabled or failed to initialize"
        })
    
    # Get statistics
    stats = {
        "enabled": True,
        "using_neo4j": kg_builder.use_neo4j,
        "stats": {}
    }
    
    # Add in-memory stats if applicable
    if not kg_builder.use_neo4j:
        stats["stats"]["entity_count"] = len(kg_builder.in_memory_graph["entities"])
        stats["stats"]["relationship_count"] = len(kg_builder.in_memory_graph["relationships"])
    else:
        # Get Neo4j stats
        try:
            entity_count = kg_builder.graph.query("MATCH (n) RETURN count(n) as count")
            relationship_count = kg_builder.graph.query("MATCH ()-[r]->() RETURN count(r) as count")
            
            stats["stats"]["entity_count"] = entity_count[0]["count"] if entity_count else 0
            stats["stats"]["relationship_count"] = relationship_count[0]["count"] if relationship_count else 0
        except Exception as e:
            stats["stats"]["error"] = str(e)
    
    return jsonify(stats)

@app.route("/kg_entities", methods=["GET"])
def get_kg_entities():
    """Endpoint to get knowledge graph entities for a video"""
    video_id = request.args.get("video_id")
    
    if not video_id:
        return jsonify({"error": "Missing video_id parameter"}), 400
        
    if not USE_KNOWLEDGE_GRAPH or not kg_builder:
        return jsonify({
            "enabled": False,
            "message": "Knowledge graph is not enabled or failed to initialize"
        })
    
    # Get entities and relationships for the video
    if kg_builder.use_neo4j:
        try:
            # Use Neo4j to query by video ID (requires metadata tagging during extraction)
            return jsonify({"error": "Neo4j entity querying by video ID not yet implemented"}), 501
        except Exception as e:
            return jsonify({"error": f"Error querying Neo4j: {str(e)}"}), 500
    else:
        # Just return the entire in-memory graph for now
        # In a real implementation, we would tag entities with video_id during extraction
        return jsonify({
            "entities": list(kg_builder.in_memory_graph["entities"].values()),
            "relationships": kg_builder.in_memory_graph["relationships"]
        })

@app.route("/delete_video", methods=["POST"])
def delete_video():
    """Endpoint to delete a processed video and all its data"""
    data = request.json
    video_id = data.get("video_id")
    
    if not video_id:
        return jsonify({"error": "Missing video_id parameter"}), 400
    
    try:
        # Delete all chunks for this video
        collection.delete(
            where={"video_id": video_id}
        )
        
        # Delete knowledge graph data if enabled
        if USE_KNOWLEDGE_GRAPH and kg_builder:
            try:
                if kg_builder.use_neo4j:
                    # In Neo4j, we would need to tag entities with video_id
                    # This is a placeholder - in a real implementation, you would delete
                    # all entities and relationships tagged with this video_id
                    pass
                else:
                    # For in-memory, since we don't have tagging yet, we won't delete KG data
                    # This would need to be implemented with proper tagging
                    pass
            except Exception as e:
                print(f"Warning: Could not delete knowledge graph data: {e}")
        
        return jsonify({"success": True, "message": f"Video {video_id} deleted successfully"})
    except Exception as e:
        print(f"Error deleting video: {str(e)}")
        return jsonify({"error": f"Failed to delete video: {str(e)}"}), 500

if __name__ == "__main__":
    if DEBUG_MODE:
        print(f"Starting server with Ollama host set to: {OLLAMA_HOST}")
        print(f"Testing Ollama connection before startup:")
        test_results = test_ollama_connection()
        print(f"Connection test results: {json.dumps(test_results, indent=2)}")
    
    # Create directories if they don't exist
    if not os.path.exists("templates"):
        os.makedirs("templates")
        
    if not os.path.exists("static"):
        os.makedirs("static")
        
    if not os.path.exists("templates/index.html"):
        with open("templates/index.html", "w") as f:
            f.write("")
    
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=DEBUG_MODE, port=port)