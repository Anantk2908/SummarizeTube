import requests
import socket
import time
import os

# IP addresses to try (in order of likelihood)
IP_ADDRESSES = [
    "10.255.255.254",  # Your current setting
    "host.docker.internal",  # Special DNS name that WSL2 can use to reach Windows
    "172.17.0.1",  # Another common WSL2->Windows IP
    "192.168.1.1",  # Common router IP (fallback if WSL is going through the router)
    "localhost"  # Try this as a sanity check
]

def test_connection(host):
    print(f"\n🔍 Testing connection to Ollama at {host}:11434")
    
    # Test socket connection first
    try:
        print(f"  Socket test: ", end="", flush=True)
        start = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, 11434))
        duration = time.time() - start
        
        if result == 0:
            print(f"✅ Success ({duration:.2f}s)")
            sock.close()
        else:
            print(f"❌ Failed - Port closed or unreachable ({duration:.2f}s)")
            sock.close()
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    
    # Test HTTP API
    try:
        print(f"  HTTP API test: ", end="", flush=True)
        start = time.time()
        response = requests.get(f"http://{host}:11434/api/tags", timeout=5)
        duration = time.time() - start
        
        if response.status_code == 200:
            print(f"✅ Success ({duration:.2f}s)")
            models = response.json().get("models", [])
            if models:
                print(f"  📋 Available models: {', '.join(m['name'] for m in models)}")
            else:
                print(f"  ⚠️ No models found")
            return True
        else:
            print(f"❌ Failed - Status code: {response.status_code} ({duration:.2f}s)")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

# Additional test: Try Ollama client directly
def test_ollama_client(host):
    print(f"\n🤖 Testing Ollama Python client with host {host}")
    try:
        import ollama
        print("  Client test: ", end="", flush=True)
        
        # Set the OLLAMA_HOST environment variable
        original_ollama_host = os.environ.get("OLLAMA_HOST", "")
        os.environ["OLLAMA_HOST"] = f"http://{host}:11434"
        
        start = time.time()
        try:
            response = ollama.list()  # Use environment variable instead of host parameter
            duration = time.time() - start
            print(f"✅ Success ({duration:.2f}s)")
            print(f"  📋 Models via client: {', '.join(m['name'] for m in response['models'])}")
            
            # Restore original OLLAMA_HOST
            if original_ollama_host:
                os.environ["OLLAMA_HOST"] = original_ollama_host
            else:
                del os.environ["OLLAMA_HOST"]
                
            return True
        except Exception as client_err:
            duration = time.time() - start
            print(f"❌ Error: {str(client_err)} ({duration:.2f}s)")
            
            # Try direct API call as fallback
            print("  Trying direct API call as fallback: ", end="", flush=True)
            try:
                response = requests.post(
                    f"http://{host}:11434/api/chat",
                    json={"model": "llama3", "messages": [{"role": "user", "content": "hello"}]},
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"✅ Success")
                    return True
                else:
                    print(f"❌ Failed - Status: {response.status_code}")
                    return False
            except Exception as fallback_err:
                print(f"❌ Failed - {str(fallback_err)}")
                return False
            finally:
                # Restore original OLLAMA_HOST
                if original_ollama_host:
                    os.environ["OLLAMA_HOST"] = original_ollama_host
                else:
                    del os.environ["OLLAMA_HOST"]
    except ImportError:
        print("  ❌ ollama package not installed correctly")
        return False
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

# Try all IP addresses
print("🔄 Testing Ollama connectivity from WSL to Windows")
print("🔄 Make sure Ollama is running on Windows!")

success = False
working_host = None

for host in IP_ADDRESSES:
    if test_connection(host):
        print(f"\n✅ SUCCESS: Connected to Ollama at {host}:11434")
        success = True
        working_host = host
        break
    else:
        print(f"\n❌ Failed to connect to {host}:11434")

if success:
    # Try the client too
    test_ollama_client(working_host)
    
    print(f"\n🎉 CONNECTION SUCCESSFUL!")
    print(f"Update your app.py with:")
    print(f"OLLAMA_HOST = \"{working_host}\"")
    print(f"And update ollama calls to use environment variables:")
    print(f"import os")
    print(f"os.environ[\"OLLAMA_HOST\"] = f\"http://{working_host}:11434\"")
    print(f"response = ollama.chat(model=model_name, messages=[...])")
    print(f"# Don't use the host parameter")
else:
    print("\n❌ All connection attempts failed!")
    print("\nTroubleshooting tips:")
    print("1. Verify Ollama is running on Windows (check Task Manager)")
    print("2. Check Windows Firewall settings - add exception for port 11434")
    print("3. Try manually accessing http://localhost:11434 in a Windows browser")
    print("4. Restart WSL with: 'wsl --shutdown' then reopen your terminal")