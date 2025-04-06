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

# Load environment variables from .env file
load_dotenv()

# Configure Ollama from environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
DEBUG_MODE = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
# Streaming has been removed
ENABLE_STREAMING = False

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

# Initialize reranker (optional)
try:
    from sentence_transformers import CrossEncoder
    print("Initializing cross-encoder reranker...")
    # Try with a more reliably available model
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-4-v2', max_length=512)
    USE_RERANKER = True
    print("Reranker initialized successfully!")
except Exception as e:
    print(f"Warning: Could not initialize reranker: {str(e)}")
    print("Will use simpler retrieval method without reranking")
    USE_RERANKER = False

# Initialize Whisper model
whisper_model = None  # Lazy loading to save memory

# Initialize text splitter with very small chunks and more overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Very small chunks
    chunk_overlap=300,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Function to extract video ID or playlist ID from URL
def extract_video_id(youtube_url):
    """
    Extract video ID or playlist ID from various YouTube URL formats.
    Returns a tuple: (type, id) where type is 'video' or 'playlist'.
    Returns (None, None) if the URL is invalid or not recognized.

    Handles:
    - youtube.com/watch?v=...
    - youtu.be/...
    - youtube.com/shorts/...
    - youtube.com/embed/...
    - youtube.com/v/...
    - youtube.com/playlist?list=...
    """
    if not youtube_url:
        return None, None
        
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
        
        # Get query parameters
        query_params = parse_qs(parsed_url.query)
        
        # First check if it's a playlist URL
        if 'list' in query_params and parsed_url.netloc in ['youtube.com', 'www.youtube.com']:
            playlist_id = query_params['list'][0]
            # Basic validation - playlist IDs are typically about 30+ characters with alpha and numeric
            if len(playlist_id) > 10:
                print(f"Detected playlist ID: {playlist_id}")
                return 'playlist', playlist_id

        # Then handle video URLs
        # Handle youtu.be URLs (always videos)
        if parsed_url.netloc == 'youtu.be':
            video_id = parsed_url.path.strip('/')
            if video_id:
                return 'video', video_id
            
        # Handle various youtube.com formats
        if parsed_url.netloc in ['youtube.com', 'www.youtube.com']:
            # Handle /watch URLs
            if parsed_url.path == '/watch':
                if 'v' in query_params:
                    video_id = query_params['v'][0]
                    if video_id:
                        return 'video', video_id
                
            # Handle /shorts/, /embed/, and /v/ URLs
            for path_prefix in ['/shorts/', '/embed/', '/v/']:
                if parsed_url.path.startswith(path_prefix):
                    video_id = parsed_url.path.replace(path_prefix, '').split('/')[0]
                    if video_id:
                        return 'video', video_id
            
            # Handle /playlist URLs explicitly
            if parsed_url.path == '/playlist' and 'list' in query_params:
                playlist_id = query_params['list'][0]
                if len(playlist_id) > 10:
                    print(f"Detected playlist ID from /playlist path: {playlist_id}")
                    return 'playlist', playlist_id

        print(f"URL not recognized as valid YouTube video or playlist: {youtube_url}")
        return None, None
        
    except Exception as e:
        print(f"Error extracting ID from URL: {str(e)}")
        return None, None

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

# Function to get the current timestamp in different formats
def get_current_timestamp():
    now = datetime.now()
    return {
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "unix": int(now.timestamp())
    }

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
        # Check if any existing entries for this video
        try:
            # First check if any chunks exist
            result = collection.get(
                where={"video_id": {"$eq": video_id}},
                limit=1
            )
            if result and result["ids"]:
                print(f"Found existing chunks for video {video_id}, deleting them...")
                collection.delete(
                    where={"video_id": {"$eq": video_id}}
                )
            else:
                print(f"No existing chunks found for video {video_id}, skipping deletion")
        except Exception as e:
            print(f"Warning when checking/deleting existing chunks: {str(e)}")
            # Continue anyway
        
        # Split transcript into chunks for better RAG
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
        
        return True, len(chunks)
    except Exception as e:
        print(f"Error processing transcript: {str(e)}")
        return False, str(e)

# Add a robust query function before check_ollama_available() to handle potential ChromaDB errors
def safe_chroma_query(query_text, filter_condition, n_results=10):
    """
    Perform a ChromaDB query with robust error handling and fallbacks.
    Returns a tuple of (documents, metadatas, distances) or (None, None, None) if completely failed.
    """
    try:
        # First try with standard query
        results = collection.query(
            query_texts=[query_text],
            where=filter_condition,
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        if results["documents"] and len(results["documents"][0]) > 0:
            return results["documents"][0], results["metadatas"][0], results["distances"][0] if "distances" in results else None
    except Exception as e:
        print(f"Standard ChromaDB query failed: {str(e)}. Trying fallbacks...")
    
    # If standard query fails, try with smaller result set
    try:
        results = collection.query(
            query_texts=[query_text],
            where=filter_condition,
            n_results=min(5, n_results),  # Try with fewer results
            include=["metadatas", "documents", "distances"]
        )
        
        if results["documents"] and len(results["documents"][0]) > 0:
            print("Fallback query with smaller result set succeeded")
            return results["documents"][0], results["metadatas"][0], results["distances"][0] if "distances" in results else None
    except Exception as e:
        print(f"Smaller query fallback failed: {str(e)}. Trying get all...")
    
    # If all queries fail, fall back to just getting all documents with the filter
    try:
        all_results = collection.get(
            where=filter_condition,
            include=["metadatas", "documents"]
        )
        
        if all_results["documents"] and len(all_results["documents"]) > 0:
            print(f"Query fallback: using get() with {len(all_results['documents'])} documents")
            # Take the first n_results documents (or all if fewer)
            count = min(n_results, len(all_results["documents"]))
            return all_results["documents"][:count], all_results["metadatas"][:count], None
    except Exception as e:
        print(f"All ChromaDB query fallbacks failed: {str(e)}")
    
    return None, None, None

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

# Helper function to process a single video ID
def _process_single_video(video_id, youtube_url, model_name):
    """Internal function to process a single video.
       Returns a dictionary with processing results or an error.
    """
    
    # Check for Ollama and model availability (can be done once per request)
    # Moved outside this function for efficiency in playlist processing

    # Get video info (title)
    video_title = f"Video {video_id}"
    try:
        with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            video_title = info.get('title', video_title)
    except Exception as e:
        print(f"Warning: Could not fetch title for {video_id}: {str(e)}")
        # Continue processing even if title fetch fails

    # 1. Try YouTube API transcript
    transcript, timestamps = get_youtube_transcript(video_id)
    source = "youtube_api"

    # 2. If API fails, use Whisper
    if not transcript:
        print(f"YouTube API transcript failed for {video_id}, trying Whisper...")
        transcript, timestamps, whisper_result = transcribe_video(video_id)
        source = "whisper"
        # Check if whisper returned an error string
        if isinstance(whisper_result, str) and whisper_result.startswith("Error"):
            return {"success": False, "video_id": video_id, "title": video_title, "error": whisper_result}
        # If whisper returned a title, use it (might be more accurate)
        if isinstance(whisper_result, str) and whisper_result and not whisper_result.startswith("Error"):
            video_title = whisper_result 
    
    if not transcript or not timestamps:
        return {"success": False, "video_id": video_id, "title": video_title, "error": "Failed to obtain transcript via API or Whisper"}
    
    # 3. Process transcript for RAG (chunking, embedding, storing)
    success, result = process_transcript_for_rag(video_id, transcript, timestamps, video_title, youtube_url)
    if not success:
        return {"success": False, "video_id": video_id, "title": video_title, "error": f"Failed to process transcript: {result}"}

    return {
        "success": True,
        "video_id": video_id,
        "title": video_title,
        "chunks": result, # Number of chunks created
        "source": source
    }

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
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    url_type, object_id = extract_video_id(url)

    if url_type is None:
        return jsonify({"error": "Invalid YouTube URL"}), 400
    
    if url_type == 'playlist':
        # Indicate that the playlist endpoint should be used
        return jsonify({"error": "This is a playlist URL. Please use the /process_playlist endpoint."}), 400

    # It's a video URL
    video_id = object_id
    # Construct the standard YouTube video URL for consistency, even if input was different format
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    # Check for Ollama availability ONCE before processing
    if not check_ollama_available():
        return jsonify({"error": f"Ollama is not running at {OLLAMA_HOST}. Please start Ollama server."}), 500

    # Check if the requested model is available ONCE
    model_name = data.get("model", OLLAMA_MODEL)
    if not check_model_available(model_name):
        return jsonify({
            "error": f"Model '{model_name}' not found in Ollama. Please run 'ollama pull {model_name}' first."
        }), 400
    
    # Call the refactored processing function for the single video
    result = _process_single_video(video_id, youtube_url, model_name)

    if result["success"]:
        # Return success details
        return jsonify({
                "message": "Video processed successfully",
                "video_id": result["video_id"],
                "title": result["title"],
                "chunks": result["chunks"],
                "source": result["source"]
            })
    else:
        # Return error details, including video info if available
        error_details = {"error": result["error"]}
        if "video_id" in result: error_details["video_id"] = result["video_id"]
        if "title" in result: error_details["title"] = result["title"]
        return jsonify(error_details), 500

@app.route("/process_playlist", methods=["POST"])
def process_playlist():
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    print(f"Received playlist URL: {url}")
    url_type, object_id = extract_video_id(url)
    print(f"URL type: {url_type}, Object ID: {object_id}")

    if url_type is None:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    if url_type == 'video':
        # If a video URL is sent here, inform the user to use the correct endpoint
        print(f"URL was detected as video, not playlist: {url}")
        return jsonify({"error": "This is a video URL. Please use the /process_video endpoint."}), 400

    # It's a playlist URL
    playlist_id = object_id
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
    print(f"Processing playlist URL: {playlist_url}")

    # Check for Ollama availability ONCE before processing
    if not check_ollama_available():
        return jsonify({"error": f"Ollama is not running at {OLLAMA_HOST}. Please start Ollama server."}), 500
    
    # Check if the requested model is available ONCE
    model_name = data.get("model", OLLAMA_MODEL)
    if not check_model_available(model_name):
        return jsonify({
            "error": f"Model '{model_name}' not found in Ollama. Please run 'ollama pull {model_name}' first."
        }), 400

    try:
        print(f"Fetching playlist info for: {playlist_url}")
        # Use yt-dlp to get playlist entries without downloading media
        ydl_opts = {
            'extract_flat': True,  # Don't extract info for each video, just list them
            'quiet': True,
            'no_warnings': True,
            'force_generic_extractor': False # Ensure it uses youtube extractor
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)

        if 'entries' not in playlist_info or not playlist_info['entries']:
            return jsonify({"error": "Could not find any videos in the playlist or playlist is private/invalid."}), 404

        playlist_title = playlist_info.get('title', f"Playlist {playlist_id}")
        video_entries = playlist_info['entries']
        total_videos = len(video_entries)
        print(f"Found {total_videos} videos in playlist '{playlist_title}'")

        # First delete any existing entries for this playlist
        try:
            # First check if any chunks exist
            result = collection.get(
                where={"playlist_id": {"$eq": playlist_id}},
                limit=1
            )
            if result and result["ids"]:
                print(f"Found existing chunks for playlist {playlist_id}, deleting them...")
                collection.delete(
                    where={"playlist_id": {"$eq": playlist_id}}
                )
            else:
                print(f"No existing chunks found for playlist {playlist_id}, skipping deletion")
        except Exception as e:
            print(f"Warning when checking/deleting existing playlist chunks: {str(e)}")
            # Continue anyway

        processed_count = 0
        failed_count = 0
        results_summary = []
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        # Process each video in the playlist
        for i, entry in enumerate(video_entries):
            video_id = entry.get('id')
            video_title_in_playlist = entry.get('title', f"Video {video_id}") # Title from playlist entry
            video_idx = i + 1  # 1-based index in playlist
            
            # Get duration in seconds and format it
            duration_seconds = entry.get('duration')
            formatted_duration = "??:??"
            if duration_seconds:
                # Convert seconds to HH:MM:SS format
                hours, remainder = divmod(int(duration_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            if not video_id:
                print(f"Skipping entry {video_idx}/{total_videos}: Missing video ID.")
                failed_count += 1
                results_summary.append({"status": "skipped", "reason": "Missing video ID", "entry_index": video_idx})
                continue
                
            # Construct the standard video URL for processing consistency
            standard_video_url = f"https://www.youtube.com/watch?v={video_id}"

            print(f"Processing video {video_idx}/{total_videos}: {video_id} ({video_title_in_playlist[:50]}...)")
            
            # Get video transcript
            transcript, timestamps = get_youtube_transcript(video_id)
            source = "youtube_api"
            
            # If YouTube API failed, try Whisper
            if not transcript:
                print(f"YouTube API transcript failed for {video_id}, trying Whisper...")
                transcript, timestamps, whisper_result = transcribe_video(video_id)
                source = "whisper"
                if not transcript or not timestamps:
                    print(f"Failed to get transcript for video {video_id}")
                    failed_count += 1
                    results_summary.append({
                        "status": "failed", 
                        "video_id": video_id,
                        "title": video_title_in_playlist,
                        "error": "Failed to obtain transcript"
                    })
                    continue
            
            # Process transcript
            try:
                # Split transcript into chunks
                chunks = text_splitter.split_text(transcript)
                
                # Calculate positions for timestamps
                total_len = len(transcript)
                curr_pos = 0
                chunk_positions = []
                
                # Calculate starting positions of each chunk in the full text
                for chunk in chunks:
                    chunk_positions.append(curr_pos / total_len)
                    curr_pos += len(chunk)
                
                # Store chunks with metadata
                for j, chunk in enumerate(chunks):
                    position = chunk_positions[j]
                    
                    # Find closest timestamp for this chunk
                    matching_timestamp = None
                    for ts in timestamps:
                        ts_pos = ts["start"] / timestamps[-1]["start"] if timestamps[-1]["start"] > 0 else 0
                        if matching_timestamp is None or abs(ts_pos - position) < abs(matching_timestamp["start"] / timestamps[-1]["start"] - position):
                            matching_timestamp = ts
                    
                    # Generate unique ID for this chunk
                    chunk_id = f"{playlist_id}_video_{video_id}_chunk_{j}"
                    
                    # Add to collections for batch insertion
                    all_ids.append(chunk_id)
                    all_metadatas.append({
                        "video_id": video_id,
                        "playlist_id": playlist_id,
                        "playlist_title": playlist_title,
                        "video_title": video_title_in_playlist,
                        "video_index": video_idx,
                        "url": standard_video_url,
                        "playlist_url": playlist_url,
                        "chunk_id": j,
                        "timestamp": matching_timestamp["start"] if matching_timestamp else 0,
                        "formatted_time": matching_timestamp["formatted_time"] if matching_timestamp else "00:00:00",
                        "duration": formatted_duration,
                        "duration_seconds": duration_seconds,
                        "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                        "source": source,
                        "is_playlist_chunk": True
                    })
                    all_chunks.append(chunk)
                
                processed_count += 1
                results_summary.append({
                    "status": "success", 
                    "video_id": video_id,
                    "title": video_title_in_playlist,
                    "chunks": len(chunks),
                    "index": video_idx,
                    "source": source
                })
                
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
                failed_count += 1
                results_summary.append({
                    "status": "failed", 
                                "video_id": video_id,
                    "title": video_title_in_playlist,
                    "error": str(e)
                })
        
        # Insert all chunks in one batch operation
        if all_chunks:
            print(f"Inserting {len(all_chunks)} chunks into ChromaDB for playlist {playlist_id}")
            collection.add(
                ids=all_ids,
                metadatas=all_metadatas,
                documents=all_chunks
            )
        
        # Return summary of playlist processing
        return jsonify({
            "message": f"Playlist processing complete for '{playlist_title}'",
            "playlist_id": playlist_id,
            "playlist_title": playlist_title,
            "total_videos_in_playlist": total_videos,
            "successfully_processed": processed_count,
            "failed_or_skipped": failed_count,
            "total_chunks": len(all_chunks),
            "results": results_summary
        })

    except yt_dlp.utils.DownloadError as e:
         # Handle specific yt-dlp errors like private playlists or invalid URLs
        print(f"yt-dlp error processing playlist {playlist_url}: {str(e)}")
        return jsonify({"error": f"Failed to fetch playlist information. It might be private, invalid, or unavailable. Error: {str(e)}"}), 404
    except Exception as e:
        print(f"Error processing playlist {playlist_url}: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"An unexpected error occurred processing the playlist: {str(e)}"}), 500

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

SPECIAL INSTRUCTION FOR TIME REFERENCES:
When referring to a specific moment in the video in your answer, ALWAYS format it as [HH:MM:SS] (e.g., [01:24:30]).
This exact format is crucial as it will be automatically detected and converted to clickable links for viewers.
Examples of proper time formatting: [00:05:10], [01:15:25], [02:30:45]

TRANSCRIPT SEGMENTS:
{context}

QUESTION: {question}

ANSWER:"""

SUMMARY_PROMPT_TEMPLATE = """You are a highly skilled AI assistant specializing in creating summaries of video content.

Your task is to create a concise but comprehensive summary of a YouTube video based on the transcript segments provided below.

GUIDELINES:

1. Focus on the main topics, key points, and important details
2. Maintain chronological order of topics discussed
3. Include any significant conclusions or insights from the video
4. Keep the summary concise but informative
5. Highlight any particularly useful or unique information
6. Don't include your personal opinions or information not present in the transcript
7. Format the summary in easy-to-read paragraphs
8. Use objective, clear language

VIDEO TITLE: {title}

TRANSCRIPT SEGMENTS:
{context}

SUMMARY:"""

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    content_id = data.get("video_id")  # Can be either a video_id or playlist_id
    content_type = data.get("type", "video")  # Either "video" or "playlist"
    question = data.get("question")
    model_name = data.get("model", OLLAMA_MODEL)

    if not content_id or not question:
        return jsonify({"error": "Missing parameters"}), 400
    
    # Check if the user is asking for the current time/timestamp
    time_keywords = ["what time is it", "what's the time", "current time", "what is the time", 
                    "timestamp", "current timestamp", "what date is it", "what's the date", 
                    "current date", "what is the date"]
    
    if any(keyword in question.lower() for keyword in time_keywords):
        timestamp_data = get_current_timestamp()
        answer_text = f"The current time is {timestamp_data['time']} on {timestamp_data['date']}."
        return jsonify({
            "answer": answer_text,
            "timestamps": [],
            "time_taken": 0,
            "timestamp_data": timestamp_data
        })
    
    # Check for Ollama
    if not check_ollama_available():
        return jsonify({"error": f"Ollama is not running at {OLLAMA_HOST}. Please start Ollama server."}), 500
    
    # Check if model is available
    if not check_model_available(model_name):
        return jsonify({
            "error": f"Model '{model_name}' not found in Ollama. Please run 'ollama pull {model_name}' first."
        }), 400

    try:
        print(f"Querying ChromaDB for {'playlist' if content_type == 'playlist' else 'video'} ID: {content_id}")
        
        # Set up the where filter based on content type
        if content_type == "playlist":
            filter_condition = {"playlist_id": content_id}
            print(f"Looking for chunks with playlist_id = {content_id}")
        else:
            filter_condition = {"video_id": content_id}
            print(f"Looking for chunks with video_id = {content_id}")
        
        # First get all chunks for this video or playlist
        all_chunks = collection.get(
            where=filter_condition,
            include=["metadatas", "documents"]
        )
        
        if not all_chunks["documents"]:
            return jsonify({"error": f"No content found for this {content_type} ID"}), 404
            
        # Filter out summaries manually
        filtered_docs = []
        filtered_metadatas = []
        
        for i, metadata in enumerate(all_chunks["metadatas"]):
            if metadata.get("type") != "summary":
                filtered_docs.append(all_chunks["documents"][i])
                filtered_metadatas.append(metadata)
        
        if not filtered_docs:
            return jsonify({"error": f"No transcript chunks found for this {content_type}"}), 404
            
        print(f"Found {len(filtered_docs)} transcript chunks")
        
        # Get initial candidates using vector search
        try:
            print(f"Performing robust query for: {question}")
            filtered_results_docs, filtered_results_metadatas, filtered_results_distances = safe_chroma_query(
                question,
                filter_condition,
                n_results=10
            )
            
            if not filtered_results_docs or not filtered_results_metadatas:
                return jsonify({"error": f"No relevant transcript chunks found for this question"}), 404
                
            print(f"Query returned {len(filtered_results_docs)} relevant chunks")
                
        except Exception as e:
            print(f"Error in safe query: {str(e)}")
            return jsonify({"error": f"Error retrieving content: {str(e)}"}), 500
        
        # Perform reranking if available, otherwise use distance-based ranking
        chunks_with_scores = []
        
        if USE_RERANKER:
            # Prepare pairs for reranking
            print("Using cross-encoder reranking...")
            pairs = [(question, doc) for doc in filtered_results_docs]
            
            # Rerank the chunks
            rerank_scores = reranker.predict(pairs)
            
            # Combine chunks with their scores and metadata
            for i, doc in enumerate(filtered_results_docs):
                metadata = filtered_results_metadatas[i]
                # Extra fields for playlist chunks
                video_info = {}
                if content_type == "playlist" and metadata.get("is_playlist_chunk", False):
                    video_info = {
                        "video_id": metadata.get("video_id", ""),
                        "video_title": metadata.get("video_title", "Unknown Video"),
                        "video_index": metadata.get("video_index", 0),
                        "video_url": metadata.get("url", "")
                    }
                    
                chunks_with_scores.append({
                    "text": doc,
                    "score": float(rerank_scores[i]),
                    "timestamp": float(metadata.get("timestamp", 0)),
                    "formatted_time": metadata.get("formatted_time", "00:00:00"),
                    "text_preview": metadata.get("text_preview", ""),
                    **video_info  # Add video-specific info for playlists
                })
            
            # Sort by reranking score (higher is better)
            chunks_with_scores.sort(key=lambda x: x["score"], reverse=True)
        else:
            # Use distance-based scoring (lower distance is better)
            print("Using distance-based ranking (no reranker)...")
            for i, doc in enumerate(filtered_results_docs):
                metadata = filtered_results_metadatas[i]
                distance = filtered_results_distances[i] if filtered_results_distances else 1.0
                
                # Extra fields for playlist chunks
                video_info = {}
                if content_type == "playlist" and metadata.get("is_playlist_chunk", False):
                    video_info = {
                        "video_id": metadata.get("video_id", ""),
                        "video_title": metadata.get("video_title", "Unknown Video"),
                        "video_index": metadata.get("video_index", 0),
                        "video_url": metadata.get("url", "")
                    }
                
                chunks_with_scores.append({
                    "text": doc,
                    "score": 1.0 - float(distance),  # Convert distance to similarity score
                    "timestamp": float(metadata.get("timestamp", 0)),
                    "formatted_time": metadata.get("formatted_time", "00:00:00"),
                    "text_preview": metadata.get("text_preview", ""),
                    **video_info  # Add video-specific info for playlists
                })
            
            # Sort by distance-based score (higher is better)
            chunks_with_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Select top chunks based on scores
        selected_chunks = []
        timestamps = []
        
        # For playlists, ensure we get a diverse set of chunks from different videos
        if content_type == "playlist":
            # Group chunks by video_id
            chunks_by_video = {}
            for chunk in chunks_with_scores:
                video_id = chunk.get("video_id", "unknown")
                if video_id not in chunks_by_video:
                    chunks_by_video[video_id] = []
                chunks_by_video[video_id].append(chunk)
            
            # Select top chunks from each video (up to 2 per video, prioritizing the best scoring chunks)
            video_chunks = []
            for video_id, video_chunks_list in chunks_by_video.items():
                # Sort chunks for this video by score
                video_chunks_list.sort(key=lambda x: x["score"], reverse=True)
                # Take up to 2 best chunks from this video
                video_chunks.extend(video_chunks_list[:2])
            
            # Sort these selected chunks by score again and take the top 8 (or all if fewer)
            video_chunks.sort(key=lambda x: x["score"], reverse=True)
            selected_video_chunks = video_chunks[:8] if len(video_chunks) > 8 else video_chunks
            
            # Use these selected chunks
            for chunk in selected_video_chunks:
                # Extract video info before enhancing the chunk
                video_index = chunk.get('video_index', 0)
                video_title = chunk.get('video_title', 'Unknown')
                formatted_time = chunk.get('formatted_time', '00:00:00')
                
                # Add video info header to each chunk
                enhanced_chunk = f"[FROM VIDEO {video_index}: \"{video_title}\", TIMESTAMP: {formatted_time}]\n{chunk['text']}"
                selected_chunks.append(enhanced_chunk)
                
                timestamp_info = {
                    "time": chunk["timestamp"],
                    "formatted_time": formatted_time,
                    "text_preview": chunk["text_preview"],
                    "video_id": chunk.get("video_id", ""),
                    "video_title": video_title,
                    "video_index": video_index,
                    "video_url": chunk.get("video_url", "")
                }
                
                timestamps.append(timestamp_info)
                print(f"Selected playlist chunk with score {chunk.get('score', 0):.4f} from Video {timestamp_info['video_index']}: {chunk['text_preview'][:50]}")
        else:
            # Use top 5 chunks after reranking for single videos
            for chunk in chunks_with_scores[:5]:
                selected_chunks.append(chunk["text"])
            
                timestamp_info = {
                    "time": chunk["timestamp"],
                    "formatted_time": chunk["formatted_time"],
                    "text_preview": chunk["text_preview"]
                }
                
                timestamps.append(timestamp_info)
                print(f"Selected chunk with score {chunk.get('score', 0):.4f}: {chunk['text_preview'][:50]}")

        if not selected_chunks:
            return jsonify({"error": "Could not find relevant content for this question"}), 404
            
        print(f"Selected {len(selected_chunks)} chunks for response")

        # Combine chunks with paragraph breaks for better readability
        context = "\n\n".join(selected_chunks)
        
        # Debug output for chunk structure
        if DEBUG_MODE:
            print(f"\n=== DEBUG: Chunk Structure for {content_type} query ===")
            print(f"Total chunks selected: {len(selected_chunks)}")
            for i, chunk in enumerate(selected_chunks[:2]):  # Print first 2 chunks
                print(f"Chunk {i+1} (first 150 chars): {chunk[:150]}")
            print(f"Context length: {len(context)} characters")
            if content_type == "playlist":
                video_indices = list(set([ts.get("video_index", 0) for ts in timestamps]))
                print(f"Videos represented in chunks: {video_indices}")
            print("=== END DEBUG ===\n")
        
        # Generate answer
        start_time = time.time()
        
        # Choose the appropriate prompt template
        if content_type == "playlist":
            # Use playlist-specific prompt
            prompt = """You are a highly knowledgeable AI assistant that provides accurate, detailed answers about YouTube video playlists.

Your task is to answer questions about a playlist using ONLY the provided transcript segments. Follow these guidelines:

1. Use ONLY the information from the provided transcript segments
2. If the exact information isn't in the segments, say "This information is not in the playlist segments provided."
3. If you can make a reasonable inference from the segments, start with "Based on the context..."
4. ALWAYS specify which video in the playlist contains the information by referring to its number and title
5. If information spans multiple videos, clearly indicate each source video
6. Include relevant timestamps for each piece of information
7. Structure your answer logically, presenting information in order of relevance
8. Be precise about which video contains which specific information
9. If different videos provide contradictory information, highlight the discrepancy
10. When quoting directly, cite the exact video number

FORMATTING GUIDELINES:
- When introducing information from a specific video, start that paragraph with "In Video X:"
- Use bullet points when listing examples or features from the same video
- Use headings (with ### markdown) to separate information from different videos if appropriate
- Bold important terms or concepts
- Use a numbered list if providing sequential steps or explaining a process

SPECIAL INSTRUCTION FOR VIDEO AND TIME REFERENCES:
When referring to a specific moment in a video in your answer, ALWAYS format it as "Video X [HH:MM:SS]" (e.g., "Video 3 [01:24:30]").
This exact format is crucial as it will be automatically detected and converted to clickable links for viewers.

TRANSCRIPT SEGMENTS:
{context}

QUESTION: {question}

ANSWER:""".format(context=context, question=question)

            print(f"Sending playlist prompt to Ollama: {prompt[:200]}...")  # Print first 200 chars of prompt
            
            # Query Ollama with error handling for playlist prompt
            try:
                print(f"Calling Ollama API with model: {model_name}")
                response = ollama.chat(
                    model=model_name, 
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract relevant data from ChatResponse object
                # Check if we have a new-style ChatResponse object (dict-like) or old-style object with attributes
                if isinstance(response, dict) or hasattr(response, 'keys'):
                    # New style response (dictionary-like)
                    print("Using new-style dictionary response handling")
                    response_data = {
                        "model": response.get("model", model_name),
                        "message": response.get("message", {})
                    }
                    answer_text = response.get("message", {}).get("content", "")
                else:
                    # Old style response (object with attributes)
                    print("Using old-style object response handling")
                    response_data = {
                        "model": getattr(response, "model", model_name),
                        "created_at": getattr(response, "created_at", None),
                        "eval_duration": getattr(response, "eval_duration", None),
                        "message": {
                            "role": getattr(response.message, "role", "assistant") if hasattr(response, "message") else "assistant",
                            "content": getattr(response.message, "content", None) if hasattr(response, "message") else None
                        }
                    }
                    answer_text = getattr(response.message, "content", "") if hasattr(response, "message") else ""
                
                print(f"Ollama response data for playlist: {json.dumps(response_data, indent=2)}")
                
                # Validate response content
                if not answer_text:
                    print("ERROR: Answer text is empty")
                    print(f"Response object structure: {type(response)}")
                    if isinstance(response, dict):
                        print(f"Response keys: {response.keys()}")
                    elif hasattr(response, "__dict__"):
                        print(f"Response attributes: {response.__dict__.keys()}")
                    raise ValueError("Response missing content")
                    
                print(f"Answer text extracted (first 100 chars): {answer_text[:100]}")
                
                if not answer_text:
                    print("ERROR: Answer text is empty")
                    raise ValueError("Empty answer content")
                    
                print(f"Successfully processed playlist answer with length: {len(answer_text)}")
            except Exception as e:
                print(f"ERROR in Ollama chat for playlist: {str(e)}")
                print(f"Exception type: {type(e)}")
                if 'response' in locals():
                    print(f"Response content: {getattr(response.message, 'content', 'No content available')}")
                raise ValueError(f"Failed to get valid response from Ollama for playlist: {str(e)}")
        else:
            # Use standard video prompt
            prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)

            print(f"Sending prompt to Ollama: {prompt[:200]}...")  # Print first 200 chars of prompt

            # Query Ollama with error handling
            try:
                response = ollama.chat(
                    model=model_name, 
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract relevant data from ChatResponse object
                # Check if we have a new-style ChatResponse object (dict-like) or old-style object with attributes
                if isinstance(response, dict) or hasattr(response, 'keys'):
                    # New style response (dictionary-like)
                    print("Using new-style dictionary response handling")
                    response_data = {
                        "model": response.get("model", model_name),
                        "message": response.get("message", {})
                    }
                    answer_text = response.get("message", {}).get("content", "")
                else:
                    # Old style response (object with attributes)
                    print("Using old-style object response handling")
                    response_data = {
                        "model": getattr(response, "model", model_name),
                        "created_at": getattr(response, "created_at", None),
                        "eval_duration": getattr(response, "eval_duration", None),
                        "message": {
                            "role": getattr(response.message, "role", "assistant") if hasattr(response, "message") else "assistant",
                            "content": getattr(response.message, "content", None) if hasattr(response, "message") else None
                        }
                    }
                    answer_text = getattr(response.message, "content", "") if hasattr(response, "message") else ""
                
                print(f"Ollama response data: {json.dumps(response_data, indent=2)}")
                
                # Validate response content
                if not answer_text:
                    print("ERROR: Answer text is empty")
                    print(f"Response object structure: {type(response)}")
                    if isinstance(response, dict):
                        print(f"Response keys: {response.keys()}")
                    elif hasattr(response, "__dict__"):
                        print(f"Response attributes: {response.__dict__.keys()}")
                    raise ValueError("Response missing content")
                    
                print(f"Answer text extracted (first 100 chars): {answer_text[:100]}")
                
                if not answer_text:
                    print("ERROR: Answer text is empty")
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
            "content_type": content_type,
            "time_taken": round(end_time - start_time, 2),
            "debug_info": {
                "chunk_count": len(selected_chunks),
                "videos_used": list(set([ts.get("video_index", 0) for ts in timestamps])) if content_type == "playlist" else [],
                "model_used": model_name,
                "prompt_length": len(prompt) if DEBUG_MODE else 0,
                "answer_length": len(answer_text) if DEBUG_MODE else 0
            } if DEBUG_MODE else {}
        })
            
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        print(f"Full error details: {traceback.format_exc()}")
        
        # Prepare a more informative error message for the UI
        error_message = f"Error generating response: {str(e)}"
        if "Failed to get valid response from Ollama" in str(e):
            error_message = "The AI model failed to provide a valid response. This might be due to the complexity of the question or the context provided. Please try a different question or use a more capable model."
        
        return jsonify({
            "error": error_message,
            "timestamps": timestamps if 'timestamps' in locals() else [],
            "content_type": content_type,
            "debug_info": {
                "error_type": str(type(e)),
                "error_details": str(e),
                "traceback": traceback.format_exc() if DEBUG_MODE else ""
            } if DEBUG_MODE else {}
        }), 500

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
    
    # Query database for relevant chunks for summary using our robust method
    summary_chunks = []
    
    try:
        print(f"Performing robust query for video summary: {video_id}")
        query_docs, query_metadatas, _ = safe_chroma_query(
            "What is this video about?",
            {"video_id": video_id},
            n_results=10
        )
        
        if query_docs and query_metadatas:
            for i, metadata in enumerate(query_metadatas):
                if metadata.get("type") != "summary":
                        summary_chunks.append(query_docs[i])
        
        print(f"Query returned {len(summary_chunks)} relevant chunks for summary")
    except Exception as e:
        print(f"Error in summary query: {str(e)}")
        # Fall back to using all transcript chunks if query fails
        print(f"Falling back to using all transcript chunks for summary")
        summary_chunks = transcript_docs[:10]  # Use up to 10 chunks
    
    if not summary_chunks:
        return jsonify({"error": "No transcript chunks available for summarization"}), 404
    
    context = "\n\n".join(summary_chunks)
    
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

@app.route("/summarize_playlist", methods=["POST"])
def summarize_playlist():
    """Endpoint to generate a hierarchical summary of a playlist"""
    data = request.json
    playlist_id = data.get("playlist_id")
    model_name = data.get("model", OLLAMA_MODEL)
    force_regenerate = data.get("force_regenerate", False)

    if not playlist_id:
        return jsonify({"error": "Missing playlist_id"}), 400
    
    # Check for Ollama
    if not check_ollama_available():
        return jsonify({"error": f"Ollama is not running at {OLLAMA_HOST}. Please start Ollama server."}), 500
    
    # Check if model is available
    if not check_model_available(model_name):
        return jsonify({
            "error": f"Model '{model_name}' not found in Ollama. Please run 'ollama pull {model_name}' first."
        }), 400
    
    # First check if we already have a cached summary (unless force_regenerate is true)
    if not force_regenerate:
        try:
            cached_results = collection.get(
                where={
                    "$and": [
                        {"playlist_id": {"$eq": playlist_id}},
                        {"type": {"$eq": "playlist_summary"}}
                    ]
                },
                include=["documents", "metadatas"]
            )
            
            if cached_results["documents"] and len(cached_results["documents"]) > 0:
                # Return cached summary if it exists
                print(f"Returning cached playlist summary for {playlist_id}")
                
                metadata = cached_results["metadatas"][0]
                playlist_title = metadata.get("playlist_title", f"Playlist {playlist_id}")
                
                try:
                    # Parse the cached summary which is stored as JSON
                    summary_data = json.loads(cached_results["documents"][0])
                    summary_data["cached"] = True
                    summary_data["playlist_title"] = playlist_title
                    return jsonify(summary_data)
                except json.JSONDecodeError:
                    # If parsing fails, just return the raw text
                    return jsonify({
                        "overview": cached_results["documents"][0],
                        "cached": True,
                        "created_at": metadata.get("created_at", "unknown"),
                        "playlist_title": playlist_title
                    })
        except Exception as e:
            print(f"Error checking for cached playlist summary: {str(e)}")
            # Continue with generating a new summary

    try:
        # Get all chunks for this playlist
        all_chunks = collection.get(
            where={"playlist_id": playlist_id},
            include=["metadatas", "documents"]
        )
        
        if not all_chunks["documents"]:
            return jsonify({"error": "No content found for this playlist ID"}), 404
        
        # Group chunks by video_id to process each video separately
        videos = {}
        playlist_title = ""
        
        for i, metadata in enumerate(all_chunks["metadatas"]):
            # Skip summary chunks
            if metadata.get("type") == "summary":
                continue
                
            video_id = metadata.get("video_id")
            if not video_id:
                continue
                
            # Get playlist title from any chunk (should be same for all)
            if not playlist_title and "playlist_title" in metadata:
                playlist_title = metadata["playlist_title"]
            
            # Group by video
            if video_id not in videos:
                videos[video_id] = {
                    "id": video_id,
                    "title": metadata.get("video_title", f"Video {video_id}"),
                    "url": metadata.get("url", f"https://www.youtube.com/watch?v={video_id}"),
                    "index": metadata.get("video_index", 0),
                    "chunks": [],
                    "metadatas": []
                }
            
            videos[video_id]["chunks"].append(all_chunks["documents"][i])
            videos[video_id]["metadatas"].append(metadata)
        
        if not videos:
            return jsonify({"error": "No valid video chunks found in this playlist"}), 404
        
        # Sort videos by their index in the playlist
        sorted_videos = sorted(videos.values(), key=lambda v: v["index"])
        
        print(f"Found {len(sorted_videos)} videos to summarize in playlist")
        
        # Generate the playlist overview summary first
        # We'll sample content from each video to create a representative overview
        playlist_context = []
        
        for video in sorted_videos:
            # Include a header for each video
            playlist_context.append(f"Video {video['index']}: {video['title']}")
            
            # Sample some text from this video (first, middle, and last chunk)
            chunks = video["chunks"]
            if len(chunks) >= 3:
                playlist_context.append(chunks[0])
                playlist_context.append(chunks[len(chunks)//2])
                playlist_context.append(chunks[-1])
            else:
                # If less than 3 chunks, include all
                playlist_context.extend(chunks)
            
            # Add separator
            playlist_context.append("\n---\n")
        
        # Create the overview prompt
        overview_prompt = f"""You are tasked with creating a comprehensive overview summary of a YouTube playlist.

PLAYLIST TITLE: {playlist_title}

Below are sample excerpts from {len(sorted_videos)} videos in this playlist. Use these to understand the overall themes, purpose, and progression of the playlist.

{" ".join(playlist_context)}

Please provide:
1. A concise yet comprehensive overview of the entire playlist (200-300 words)
2. The main themes or topics covered
3. How the videos relate to each other and progress through the topics
4. The overall purpose or goal of the playlist

FORMAT YOUR RESPONSE IN HTML with appropriate headings, paragraphs, and bullet points for better readability.
"""

        # Generate overview with Ollama
        overview_response = ollama.chat(
            model=model_name, 
            messages=[{"role": "user", "content": overview_prompt}]
        )
        
        overview_html = overview_response.message.content if hasattr(overview_response.message, "content") else ""
        
        # Now generate individual video summaries
        video_summaries = []
        
        for video in sorted_videos:
            print(f"Generating summary for video {video['index']}: {video['title']}")
            
            # Combine chunks with paragraph breaks for better readability
            video_text = "\n\n".join(video["chunks"])
            
            # Create the video summary prompt
            video_prompt = f"""You are tasked with creating a concise but informative summary of a single video from a YouTube playlist.

VIDEO TITLE: {video['title']}
VIDEO POSITION: Video {video['index']} in the playlist

TRANSCRIPT CONTENT:
{video_text}

Please provide:
1. A concise summary of this specific video (100-150 words)
2. 3-5 key points or takeaways from the video

FORMAT YOUR RESPONSE AS JSON with the following structure:
{{
    "summary": "The summary text...",
    "key_points": ["Point 1", "Point 2", "Point 3"]
}}
"""

            try:
                # Generate summary with Ollama
                video_response = ollama.chat(
                    model=model_name, 
                    messages=[{"role": "user", "content": video_prompt}]
                )
                
                video_summary_text = video_response.message.content if hasattr(video_response.message, "content") else ""
                
                # Try to parse JSON response
                try:
                    summary_json = json.loads(video_summary_text)
                    summary_json["title"] = video["title"]
                    summary_json["url"] = video["url"]
                    summary_json["index"] = video["index"]
                    summary_json["thumbnail"] = f"https://img.youtube.com/vi/{video['id']}/mqdefault.jpg"
                    
                    # Try to extract duration if available
                    for metadata in video["metadatas"]:
                        if "duration" in metadata:
                            summary_json["duration"] = metadata["duration"]
                            break
                    
                    # If no duration found, try to get from yt-dlp
                    if "duration" not in summary_json:
                        try:
                            with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as ydl:
                                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video['id']}", download=False)
                                duration_seconds = info.get('duration')
                                if duration_seconds:
                                    hours, remainder = divmod(int(duration_seconds), 3600)
                                    minutes, seconds = divmod(remainder, 60)
                                    summary_json["duration"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        except Exception as e:
                            print(f"Error getting duration for video {video['id']}: {str(e)}")
                    
                    video_summaries.append(summary_json)
                except json.JSONDecodeError:
                    # If not valid JSON, use the raw text
                    fallback_summary = {
                        "title": video["title"],
                        "url": video["url"],
                        "index": video["index"],
                        "summary": video_summary_text,
                        "key_points": [],
                        "thumbnail": f"https://img.youtube.com/vi/{video['id']}/mqdefault.jpg"
                    }
                    
                    # Try to extract duration if available
                    for metadata in video["metadatas"]:
                        if "duration" in metadata:
                            fallback_summary["duration"] = metadata["duration"]
                            break
                    
                    # If no duration found, try to get from yt-dlp
                    if "duration" not in fallback_summary:
                        try:
                            with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as ydl:
                                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video['id']}", download=False)
                                duration_seconds = info.get('duration')
                                if duration_seconds:
                                    hours, remainder = divmod(int(duration_seconds), 3600)
                                    minutes, seconds = divmod(remainder, 60)
                                    fallback_summary["duration"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        except Exception as e:
                            print(f"Error getting duration for video {video['id']}: {str(e)}")
                    
                    video_summaries.append(fallback_summary)
            except Exception as e:
                print(f"Error generating summary for video {video['index']}: {str(e)}")
                # Add a placeholder if summary generation fails
                error_summary = {
                    "title": video["title"],
                    "url": video["url"],
                    "index": video["index"],
                    "summary": "Unable to generate summary for this video.",
                    "key_points": [],
                    "thumbnail": f"https://img.youtube.com/vi/{video['id']}/mqdefault.jpg"
                }
                
                # Try to extract duration if available
                for metadata in video["metadatas"]:
                    if "duration" in metadata:
                        error_summary["duration"] = metadata["duration"]
                        break
                
                # If no duration found, try to get from yt-dlp (but do it silently)
                if "duration" not in error_summary:
                    try:
                        with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as ydl:
                            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video['id']}", download=False)
                            duration_seconds = info.get('duration')
                            if duration_seconds:
                                hours, remainder = divmod(int(duration_seconds), 3600)
                                minutes, seconds = divmod(remainder, 60)
                                error_summary["duration"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    except:
                        # Just silently continue if there's an error
                        pass
                
                video_summaries.append(error_summary)
        
        # Prepare the final results
        timestamp = datetime.now().isoformat()
        summary_data = {
            "overview": overview_html,
            "overview_html": overview_html,
            "video_summaries": video_summaries,
            "playlist_title": playlist_title,
            "created_at": timestamp,
            "cached": False
        }
        
        # Store the summary in the collection for future use
        try:
            summary_json = json.dumps(summary_data)
            collection.upsert(
                ids=[f"{playlist_id}_playlist_summary"],
                metadatas=[{
                    "playlist_id": playlist_id,
                    "playlist_title": playlist_title,
                    "type": "playlist_summary",
                    "created_at": timestamp
                }],
                documents=[summary_json]
            )
            print(f"Cached playlist summary for {playlist_id}")
        except Exception as e:
            print(f"Error caching playlist summary: {str(e)}")
        
        return jsonify(summary_data)
    except Exception as e:
        print(f"Error generating playlist summary: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error generating playlist summary: {str(e)}"}), 500

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
        # Query for unique video IDs and playlists
        all_metadatas = collection.get()["metadatas"]
        processed_videos = {}
        processed_playlists = {}
        
        for metadata in all_metadatas:
            # Handle regular videos
            if "video_id" in metadata and "title" in metadata and "url" in metadata and not metadata.get("is_playlist_chunk", False):
                video_id = metadata["video_id"]
                if video_id not in processed_videos:
                    processed_videos[video_id] = {
                        "id": video_id,
                        "title": metadata["title"],
                        "url": metadata["url"],
                        "type": "video"
                    }
            
            # Handle playlist entries
            if "playlist_id" in metadata and metadata.get("is_playlist_chunk", False):
                playlist_id = metadata["playlist_id"]
                if playlist_id not in processed_playlists:
                    processed_playlists[playlist_id] = {
                        "id": playlist_id,
                        "title": metadata.get("playlist_title", f"Playlist {playlist_id}"),
                        "url": metadata.get("playlist_url", f"https://www.youtube.com/playlist?list={playlist_id}"),
                        "type": "playlist"
                    }
        
        # Combine and return both types
        all_items = list(processed_videos.values()) + list(processed_playlists.values())
        return jsonify({"videos": list(processed_videos.values()), 
                        "playlists": list(processed_playlists.values()),
                        "all": all_items})
                        
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
    """Endpoint to retrieve all chunks for a specific video or playlist"""
    content_id = request.args.get("video_id")  # Can be a video_id or playlist_id
    content_type = request.args.get("type", "video")  # Either "video" or "playlist"
    
    if not content_id:
        return jsonify({"error": "Missing ID parameter"}), 400
    
    try:
        # Set up the where filter based on content type
        if content_type == "playlist":
            filter_condition = {"playlist_id": content_id}
            print(f"Looking for chunks with playlist_id = {content_id}")
        else:
            filter_condition = {"video_id": content_id}
            print(f"Looking for chunks with video_id = {content_id}")
            
        # Get all chunks for this video/playlist
        all_chunks = collection.get(
            where=filter_condition,
            include=["metadatas", "documents"]
        )
        
        if not all_chunks["documents"]:
            return jsonify({"error": f"No chunks found for this {content_type} ID"}), 404
        
        # Prepare response with chunks and their metadata, filtering out summaries
        chunks = []
        for i, doc in enumerate(all_chunks["documents"]):
            if i < len(all_chunks["metadatas"]):
                metadata = all_chunks["metadatas"][i]
                # Only include if it's not a summary document
                if metadata.get("type") != "summary":
                    chunk_data = {
                        "id": len(chunks),  # Use new index for filtered list
                        "text": doc,
                        "timestamp": metadata.get("timestamp", 0),
                        "formatted_time": metadata.get("formatted_time", "00:00:00"),
                        "metadata": metadata
                    }
                    
                    # Add additional metadata for playlist chunks
                    if content_type == "playlist" and metadata.get("is_playlist_chunk", False):
                        chunk_data.update({
                            "video_id": metadata.get("video_id", ""),
                            "video_title": metadata.get("video_title", "Unknown Video"),
                            "video_index": metadata.get("video_index", 0),
                            "video_url": metadata.get("url", "")
                        })
                        
                    chunks.append(chunk_data)
        
        if not chunks:
            return jsonify({"error": f"No transcript chunks found for this {content_type}"}), 404
            
        # Sort chunks by video index first (for playlists) then by timestamp
        if content_type == "playlist":
            chunks.sort(key=lambda x: (x.get("video_index", 0), x["timestamp"]))
        else:
            chunks.sort(key=lambda x: x["timestamp"])
        
        return jsonify({
            "id": content_id,
            "type": content_type,
            "chunks": chunks,
            "count": len(chunks)
        })
    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error retrieving chunks: {str(e)}"}), 500

@app.route("/delete_video", methods=["POST"])
def delete_video():
    """Endpoint to delete a processed video or playlist and all its data"""
    data = request.json
    video_id = data.get("video_id")
    content_type = data.get("type", "video")
    
    if not video_id:
        return jsonify({"error": "Missing video_id parameter"}), 400
    
    try:
        # Delete based on content type (video or playlist)
        if content_type == "playlist":
            # Check if playlist chunks exist first
            result = collection.get(
                where={"playlist_id": {"$eq": video_id}},
                limit=1
            )
            if result and result["ids"]:
                # Delete all chunks for this playlist
                collection.delete(
                    where={"playlist_id": {"$eq": video_id}}
                )
                print(f"Deleted playlist {video_id} data")
            else:
                print(f"No data found for playlist {video_id}, nothing to delete")
        else:
            # Check if video chunks exist first
            result = collection.get(
                where={"video_id": {"$eq": video_id}},
                limit=1
            )
            if result and result["ids"]:
                # Delete all chunks for this video
                collection.delete(
                    where={"video_id": {"$eq": video_id}}
                )
                print(f"Deleted video {video_id} data")
            else:
                print(f"No data found for video {video_id}, nothing to delete")
        
        return jsonify({"success": True, "message": f"{content_type.capitalize()} {video_id} deleted successfully"})
    except Exception as e:
        print(f"Error deleting {content_type}: {str(e)}")
        return jsonify({"error": f"Failed to delete {content_type}: {str(e)}"}), 500

@app.route("/debug_url", methods=["POST"])
def debug_url():
    """Endpoint to debug URL parsing"""
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    url_type, object_id = extract_video_id(url)
    
    # Parse URL for more details
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        return jsonify({
            "url": url,
            "url_type": url_type,
            "object_id": object_id,
            "parsed": {
                "scheme": parsed_url.scheme,
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "query": query_params,
                "fragment": parsed_url.fragment
            },
            "is_playlist_path": parsed_url.path == '/playlist',
            "has_list_param": 'list' in query_params,
            "list_value": query_params.get('list', [None])[0]
        })
    except Exception as e:
        return jsonify({
            "url": url,
            "url_type": url_type,
            "object_id": object_id,
            "error": str(e)
        })

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