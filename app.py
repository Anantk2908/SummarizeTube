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
from sentence_transformers import CrossEncoder

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

# Initialize reranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v1')

# Initialize Whisper model
whisper_model = None  # Lazy loading to save memory

# Initialize text splitter with very small chunks and more overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=550,  # Very small chunks
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

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
        # Delete existing entries for this video
        try:
            collection.delete(ids=[f"{video_id}_chunk_{i}" for i in range(1000)])  # Attempt to delete potential old chunks
        except:
            pass  # Ignore if no entries exist
        
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
    video_id = data.get("video_id")
    question = data.get("question")
    model_name = data.get("model", OLLAMA_MODEL)

    if not video_id or not question:
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
        print(f"Querying ChromaDB for video_id: {video_id}")
        
        # First get all chunks for this video
        all_chunks = collection.get(
            where={"video_id": video_id},
            include=["metadatas", "documents"]
        )
        
        if not all_chunks["documents"]:
            return jsonify({"error": "No content found for this video ID"}), 404
            
        # Filter out summaries manually
        filtered_docs = []
        filtered_metadatas = []
        
        for i, metadata in enumerate(all_chunks["metadatas"]):
            if metadata.get("type") != "summary":
                filtered_docs.append(all_chunks["documents"][i])
                filtered_metadatas.append(metadata)
        
        if not filtered_docs:
            return jsonify({"error": "No transcript chunks found for this video"}), 404
            
        print(f"Found {len(filtered_docs)} transcript chunks for video")
        
        # Get initial candidates using vector search
        results = collection.query(
            query_texts=[question],
            n_results=10,  # Get more candidates for reranking
            include=["metadatas", "documents"]
        )
        
        print(f"Initial vector search returned {len(results['documents'][0])} chunks")
        
        # Filter out summaries from query results
        filtered_results_docs = []
        filtered_results_metadatas = []
        
        for i, metadata in enumerate(results["metadatas"][0]):
            if metadata.get("type") != "summary":
                filtered_results_docs.append(results["documents"][0][i])
                filtered_results_metadatas.append(metadata)
        
        if not filtered_results_docs:
            return jsonify({"error": "No relevant transcript chunks found for this question"}), 404
            
        print(f"After filtering, using {len(filtered_results_docs)} chunks for reranking")

        # Prepare pairs for reranking
        pairs = [(question, doc) for doc in filtered_results_docs]
        
        # Rerank the chunks
        print("Reranking chunks...")
        rerank_scores = reranker.predict(pairs)
        
        # Combine chunks with their scores and metadata
        chunks_with_scores = []
        for i, doc in enumerate(filtered_results_docs):
            metadata = filtered_results_metadatas[i]
            chunks_with_scores.append({
                "text": doc,
                "score": float(rerank_scores[i]),
                "timestamp": float(metadata.get("timestamp", 0)),
                "formatted_time": metadata.get("formatted_time", "00:00:00"),
                "text_preview": metadata.get("text_preview", "")
            })
        
        # Sort by reranking score
        chunks_with_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Select top chunks based on reranking scores
        selected_chunks = []
        timestamps = []
        
        # Use top 5 chunks after reranking
        for chunk in chunks_with_scores[:5]:
            selected_chunks.append(chunk["text"])
            timestamps.append({
                "time": chunk["timestamp"],
                "formatted_time": chunk["formatted_time"],
                "text_preview": chunk["text_preview"]
            })
            print(f"Selected chunk with score {chunk['score']:.4f}: {chunk['text_preview'][:50]}")

        if not selected_chunks:
            return jsonify({"error": "Could not find relevant content for this question"}), 404
            
        print(f"Selected {len(selected_chunks)} chunks after reranking")

        # Combine chunks with paragraph breaks for better readability
        context = "\n\n".join(selected_chunks)
        
        # Generate answer
        start_time = time.time()
        
        # Format prompt using the template
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
            "time_taken": round(end_time - start_time, 2)
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