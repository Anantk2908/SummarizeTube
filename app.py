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

# Load environment variables from .env file
load_dotenv()

# Configure Ollama from environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
DEBUG_MODE = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
# Streaming has been disabled to avoid connection issues
ENABLE_STREAMING = False  # Manually disabled to avoid connection issues

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

# Function to extract video ID from URL
def extract_video_id(youtube_url):
    if "youtube.com" in youtube_url:
        return youtube_url.split("v=")[-1].split("&")[0]
    elif "youtu.be" in youtube_url:
        return youtube_url.split("/")[-1]
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
QA_PROMPT_TEMPLATE = """You are a helpful AI that answers questions about YouTube videos.
Use ONLY the transcript segments provided below to answer the question.
If the information is not in the segments, respond with "This information is not in the video."
Do NOT include phrases like "based on the transcript" or "the video says" in your answer.
Give direct, concise answers without mentioning the source of your information.

TRANSCRIPT SEGMENTS:
{context}

QUESTION: {question}

ANSWER:"""

SUMMARY_PROMPT_TEMPLATE = """Create a concise summary of the YouTube video titled "{title}" based on these transcript excerpts.
Focus on the main points and key information.
Be direct and objective, using 3-4 paragraphs.
Do NOT mention that you're summarizing transcript excerpts.

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
        
        # Then yield the content as it streams
        yield from stream_ollama_response(model_name, prompt)
    
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
        # Query for all chunks belonging to this video
        results = collection.query(
            query_texts=[""],  # Empty query to get all chunks
            where={"video_id": video_id},
            n_results=100,  # Get more chunks, adjust as needed
            include=["metadatas", "documents"]
        )
        
        if not results["documents"] or len(results["documents"][0]) == 0:
            return jsonify({"error": "No chunks found for this video ID"}), 404
        
        # Prepare response with chunks and their metadata
        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            if i < len(results["metadatas"][0]):
                metadata = results["metadatas"][0][i]
                chunks.append({
                    "id": i,
                    "text": doc,
                    "timestamp": metadata.get("timestamp", 0),
                    "formatted_time": metadata.get("formatted_time", "00:00:00"),
                    "metadata": metadata
                })
            else:
                chunks.append({
                    "id": i,
                    "text": doc,
                    "timestamp": 0,
                    "formatted_time": "00:00:00",
                    "metadata": {}
                })
        
        # Sort chunks by timestamp
        chunks.sort(key=lambda x: x["timestamp"])
        
        return jsonify({
            "video_id": video_id,
            "chunks": chunks,
            "count": len(chunks)
        })
        
    except Exception as e:
        return jsonify({"error": f"Error retrieving chunks: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
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

    try:
        # Query Ollama (using environment variable set at the top of the file)
        start_time = time.time()
        response = ollama.chat(
            model=model_name, 
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()
        
        return jsonify({
            "answer": response["message"]["content"],
            "timestamps": timestamps,
            "time_taken": round(end_time - start_time, 2)
        })
    except Exception as e:
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

    try:
        # Generate summary with Ollama (using environment variable)
        start_time = time.time()
        response = ollama.chat(
            model=model_name, 
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()
        
        return jsonify({
            "title": video_title,
            "summary": response["message"]["content"],
            "time_taken": round(end_time - start_time, 2)
        })
    except Exception as e:
        return jsonify({"error": f"Error generating summary: {str(e)}"}), 500

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