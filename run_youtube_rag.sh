#!/bin/bash
# Script to set up and run the YouTube RAG application

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*)    echo "mac";;
        Linux*)     echo "linux";;
        CYGWIN*|MINGW*|MSYS*) echo "windows";;
        *)          echo "unknown";;
    esac
}

# Function to open URL in browser based on OS
open_browser() {
    local url=$1
    local os=$(detect_os)
    
    echo -e "${BLUE}Opening browser to $url${NC}"
    
    case "$os" in
        mac)     open "$url" ;;
        linux)   xdg-open "$url" 2>/dev/null || sensible-browser "$url" 2>/dev/null || 
                 echo -e "${YELLOW}Could not open browser automatically. Please open $url manually.${NC}" ;;
        windows) start "" "$url" 2>/dev/null || 
                 echo -e "${YELLOW}Could not open browser automatically. Please open $url manually.${NC}" ;;
        *)       echo -e "${YELLOW}Unknown OS. Please open $url manually.${NC}" ;;
    esac
}

# Check for Python
if ! command_exists python3; then
    if command_exists python; then
        PY_CMD="python"
    else
        echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
        exit 1
    fi
else
    PY_CMD="python3"
fi

echo -e "${GREEN}Using $PY_CMD${NC}"

# Check for pip
if ! command_exists pip && ! command_exists pip3; then
    echo -e "${RED}Error: pip is not installed or not in PATH${NC}"
    exit 1
fi

if command_exists pip3; then
    PIP_CMD="pip3"
else
    PIP_CMD="pip"
fi

echo -e "${GREEN}Using $PIP_CMD${NC}"

# Check if virtual environment exists and create if needed
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    $PY_CMD -m venv venv || { echo -e "${RED}Failed to create virtual environment${NC}"; exit 1; }
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
VENV_ACTIVATE="venv/bin/activate"
if [ "$(detect_os)" = "windows" ]; then
    VENV_ACTIVATE="venv/Scripts/activate"
fi

if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
else
    echo -e "${RED}Virtual environment activation script not found at $VENV_ACTIVATE${NC}"
    exit 1
fi

# Install requirements if needed
echo -e "${BLUE}Checking/installing requirements...${NC}"
if [ -f "requirements.txt" ]; then
    $PIP_CMD install -r requirements.txt || { echo -e "${RED}Failed to install requirements${NC}"; exit 1; }
else
    # Minimal requirements if requirements.txt not found
    echo -e "${YELLOW}requirements.txt not found. Installing minimal required packages...${NC}"
    $PIP_CMD install flask chromadb yt-dlp openai-whisper langchain youtube-transcript-api requests ollama python-dotenv || 
    { echo -e "${RED}Failed to install minimal requirements${NC}"; exit 1; }
fi

# Check if .env file exists, create with defaults if not
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}.env file not found. Creating with default settings...${NC}"
    cat > .env << EOF
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=30
DEBUG=False
EOF
    echo -e "${GREEN}.env file created with default settings${NC}"
fi

# Determine port (default to 5000)
PORT=${PORT:-5000}
echo -e "${BLUE}Using port $PORT${NC}"

# Run Ollama connection check
echo -e "${BLUE}Checking Ollama connection...${NC}"
if command_exists curl; then
    OLLAMA_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags || echo "failed")
    if [ "$OLLAMA_STATUS" = "200" ]; then
        echo -e "${GREEN}Ollama is running!${NC}"
    else
        echo -e "${YELLOW}Warning: Ollama does not appear to be running. Please start Ollama before using the app.${NC}"
        echo -e "${YELLOW}You can install Ollama from: https://ollama.ai/${NC}"
    fi
else
    echo -e "${YELLOW}Warning: curl not available, skipping Ollama connection check${NC}"
fi

# Run the Flask app in the background
echo -e "${GREEN}Starting Flask application...${NC}"
$PY_CMD app.py &
APP_PID=$!

# Wait for app to start (adjust timeout as needed)
echo -e "${BLUE}Waiting for application to start...${NC}"
MAX_ATTEMPTS=10
ATTEMPT=0
URL="http://localhost:$PORT"

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT+1))
    if command_exists curl; then
        if curl -s -o /dev/null "$URL"; then
            break
        fi
    elif command_exists wget; then
        if wget -q --spider "$URL"; then
            break
        fi
    else
        # No way to check, just wait longer and hope
        sleep 5
        break
    fi
    
    echo -e "${YELLOW}Waiting for server to start (attempt $ATTEMPT/$MAX_ATTEMPTS)...${NC}"
    sleep 1
    
    # Check if process is still running
    if ! kill -0 $APP_PID 2>/dev/null; then
        echo -e "${RED}Error: Flask application failed to start${NC}"
        exit 1
    fi
done

# Open browser
echo -e "${GREEN}Application is running!${NC}"
echo -e "${BLUE}URL: $URL${NC}"
open_browser "$URL"

# Information about stopping the application
echo -e "${YELLOW}To stop the application, press Ctrl+C in this terminal${NC}"

# Wait for Ctrl+C
trap "echo -e '${RED}Stopping application...${NC}'; kill $APP_PID 2>/dev/null; exit 0" INT

# Keep script running until Ctrl+C
wait $APP_PID 