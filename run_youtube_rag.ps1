# PowerShell script to set up and run the YouTube RAG application

# Function to check if a command exists
function Test-Command {
    param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        if (Get-Command $command) { return $true }
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

# Set text colors
$Red = @{ForegroundColor = "Red"}
$Green = @{ForegroundColor = "Green"}
$Yellow = @{ForegroundColor = "Yellow"}
$Blue = @{ForegroundColor = "Blue"}

# Check for Python
if (Test-Command -command "python") {
    $PythonCmd = "python"
    Write-Host "Using $PythonCmd" @Green
} else {
    Write-Host "Error: Python is not installed or not in PATH" @Red
    exit 1
}

# Check Python version
$PythonVersion = & $PythonCmd -c "import sys; print(sys.version_info.major)"
if ($PythonVersion -lt 3) {
    Write-Host "Warning: Python version is less than 3. Some features may not work." @Yellow
}

# Check for pip
if (Test-Command -command "pip") {
    $PipCmd = "pip"
    Write-Host "Using $PipCmd" @Green
} else {
    Write-Host "Error: pip is not installed or not in PATH" @Red
    exit 1
}

# Check if virtual environment exists and create if needed
if (-not (Test-Path -Path "venv")) {
    Write-Host "Virtual environment not found. Creating..." @Yellow
    & $PythonCmd -m venv venv
    if (-not $?) {
        Write-Host "Failed to create virtual environment" @Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." @Blue
$ActivateScript = ".\venv\Scripts\Activate.ps1"
if (Test-Path -Path $ActivateScript) {
    . $ActivateScript
} else {
    Write-Host "Virtual environment activation script not found at $ActivateScript" @Red
    exit 1
}

# Install requirements if needed
Write-Host "Checking/installing requirements..." @Blue
if (Test-Path -Path "requirements.txt") {
    & pip install -r requirements.txt
    if (-not $?) {
        Write-Host "Failed to install requirements" @Red
        exit 1
    }
} else {
    # Minimal requirements if requirements.txt not found
    Write-Host "requirements.txt not found. Installing minimal required packages..." @Yellow
    & pip install flask chromadb yt-dlp openai-whisper langchain youtube-transcript-api requests ollama python-dotenv
    if (-not $?) {
        Write-Host "Failed to install minimal requirements" @Red
        exit 1
    }
}

# Check if .env file exists, create with defaults if not
if (-not (Test-Path -Path ".env")) {
    Write-Host ".env file not found. Creating with default settings..." @Yellow
    @"
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=30
DEBUG=False
"@ | Out-File -FilePath ".env" -Encoding utf8
    Write-Host ".env file created with default settings" @Green
}

# Determine port (default to 5000)
$PORT = 5000
if ($env:PORT) {
    $PORT = $env:PORT
}
Write-Host "Using port $PORT" @Blue

# Run Ollama connection check
Write-Host "Checking Ollama connection..." @Blue
try {
    $Response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -ErrorAction SilentlyContinue
    if ($Response.StatusCode -eq 200) {
        Write-Host "Ollama is running!" @Green
    }
} catch {
    Write-Host "Warning: Ollama does not appear to be running. Please start Ollama before using the app." @Yellow
    Write-Host "You can install Ollama from: https://ollama.ai/" @Yellow
}

# Start the Flask app in a new process
Write-Host "Starting Flask application..." @Green
$FlaskProcess = Start-Process -FilePath $PythonCmd -ArgumentList "app.py" -PassThru -NoNewWindow

# Wait for app to start
Write-Host "Waiting for application to start..." @Blue
$MaxAttempts = 10
$Attempt = 0
$Url = "http://localhost:$PORT"

while ($Attempt -lt $MaxAttempts) {
    $Attempt++
    try {
        $TestConnection = Invoke-WebRequest -Uri $Url -Method HEAD -ErrorAction SilentlyContinue
        if ($TestConnection.StatusCode -eq 200) {
            break
        }
    } catch {
        Write-Host "Waiting for server to start (attempt $Attempt/$MaxAttempts)..." @Yellow
        Start-Sleep -Seconds 1
    }
    
    # Check if process is still running
    if ($FlaskProcess.HasExited) {
        Write-Host "Error: Flask application failed to start" @Red
        exit 1
    }
}

# Open browser
Write-Host "Application is running!" @Green
Write-Host "URL: $Url" @Blue
Start-Process $Url

# Information about stopping the application
Write-Host "To stop the application, press Ctrl+C in this terminal or close this window." @Yellow
Write-Host "Process ID: $($FlaskProcess.Id)" @Yellow

# Keep PowerShell window open to allow the Flask app to continue running
try {
    Wait-Process -Id $FlaskProcess.Id
} catch {
    # Handle Ctrl+C or other interruptions
    if (-not $FlaskProcess.HasExited) {
        Write-Host "Stopping application..." @Red
        Stop-Process -Id $FlaskProcess.Id -Force
    }
} 