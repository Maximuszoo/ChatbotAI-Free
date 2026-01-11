# Voice Chatbot - AI English Practice Assistant 🎙️🤖

A fully local voice chatbot with two conversation modes: **Classic Chat** and **Live Mode**. Built to help practice English conversation skills through natural AI interactions.

## About This Project 💡

I created this chatbot as a personal tool to:
- **Practice my English** through realistic conversations
- **Learn more** about AI integration, speech processing, and UI development
- Experiment with real-time streaming and voice activity detection
---
**Normal mode**
<img width="1082" height="799" alt="Captura de pantalla_20260106_172859" src="https://github.com/user-attachments/assets/0030e384-f9c8-47ee-aa90-2c4a955bc27b" />

https://github.com/user-attachments/assets/56a0d0cb-73ae-42df-8c5a-3f0938419d29

**Live mode**
<img width="1083" height="799" alt="Captura de pantalla_20260106_172956" src="https://github.com/user-attachments/assets/33e8d5dc-4310-4248-ba22-4b16085958b9" />

https://github.com/user-attachments/assets/c33bd6a7-a4ae-48cf-89c5-c72d019a0d53

---

## Features ✨

### 🎯 Dual Mode Interface
- **Classic Chat Mode**: Traditional message-by-message conversation with text input and voice recording
- **Live Mode**: Continuous hands-free conversation with real-time barge-in (interrupt the AI anytime by speaking)

### 🗣️ Voice Capabilities
- Real-time speech-to-text (Whisper)
- Natural text-to-speech output (Kokoro ONNX)
- Voice Activity Detection (VAD) for automatic silence detection

### 🧠 AI Features
- Local LLM conversations powered by Ollama (Llama, Mistral, etc.)
- **Streaming responses** - See and hear AI responses as they're generated
- Conversation history with context awareness

### 🎨 Modern UI
- Dark theme inspired by Google Gemini
- Responsive chat bubbles
- Customizable font sizes (small, medium, large)
- Live Mode with pulsing visual indicator

### ⚡ Smart Features
- Barge-in detection - Interrupt the AI naturally by speaking
- User preferences saved locally (model, font size, auto-send mode)
- Multi-threaded for smooth performance

## Installation 📦

### Prerequisites

**All Systems:**
- **Python 3.10 or 3.11**
- **Ollama** - Download from [ollama.ai](https://ollama.ai)

### Linux Installation 🐧

1. **Install system dependencies:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip portaudio19-dev

# Arch Linux
sudo pacman -S python python-pip portaudio
```

2. **Clone or download this repository:**
```bash
cd ~/your-projects-folder
git clone <your-repo-url>
cd ChatbotAI-English
```

3. **Create virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate
```

4. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

5. **Download required model files:**

⚠️ **Important**: These files are too large for GitHub (310+ MB) and must be downloaded separately.

- **Kokoro TTS Model** (`kokoro-v0_19.onnx`):
  - Download from: [Kokoro-82M releases](https://github.com/thewh1teagle/kokoro-onnx/releases)
  - Place in the project root folder
```bash
wget https://huggingface.co/thewh1teagle/Kokoro/resolve/main/kokoro-v0_19.onnx
```

- **Voice Configurations** (`voices.json`):
  - Download from: [Kokoro-82M releases](https://github.com/thewh1teagle/kokoro-onnx/releases)
  - Place in the project root folder
```bash
wget https://huggingface.co/thewh1teagle/Kokoro/resolve/main/voices.json
```

Your folder structure should look like:
```
ChatbotAI-English/
├── main.py
├── ai_manager.py
├── ...
├── kokoro-v0_19.onnx    ← Download this
└── voices.json           ← Download this
```

6. **Install an Ollama model:**
```bash
ollama pull llama3.1:8b
# or try other models: mistral, gemma2, etc.
```

7. **Run the application:**
```bash
python main.py
```

### Windows Installation 🪟

1. **Install Python:**
   - Download Python 3.11 from [python.org](https://www.python.org/downloads/)
   - ✅ **Important**: Check "Add Python to PATH" during installation

2. **Install Ollama:**
   - Download from [ollama.ai](https://ollama.ai)
   - Run the installer

3. **Download this project:**
   - Download as ZIP or clone with Git
   - Extract to a folder like `C:\Users\YourName\ChatbotAI-English`

4. **Open Command Prompt in the project folder:**
   - Navigate to the folder in File Explorer
   - Type `cmd` in the address bar and press Enter

5. **Create virtual environment (recommended):**
```cmd
python -m venv venv
venv\Scripts\activate
```

6. **Install dependencies:**
```cmd
pip install -r requirements.txt
```

7. **Download required model files:**

⚠️ **Important**: These files are too large for GitHub and must be downloaded separately.

- **Kokoro TTS Model** (`kokoro-v0_19.onnx`):
  - Download from: [Kokoro-82M releases](https://github.com/thewh1teagle/kokoro-onnx/releases)
  - Place in the project folder

- **Voice Configurations** (`voices.json`):
  - Download from: [Kokoro-82M releases](https://github.com/thewh1teagle/kokoro-onnx/releases)
  - Place in the project folder

8. **Install an Ollama model:**
```cmd
ollama pull llama3.1:8b
```

9. **Run the application:**
```cmd
python main.py
```

## Usage 🚀

### Classic Chat Mode
1. Type messages or use the 🎤 microphone button
2. Press ⏹ while recording to send
3. Choose between auto-send or manual review mode in settings (⚙️)

### Live Mode (Continuous Conversation)
1. Click the ✨ button to enter Live Mode
2. Speak naturally - the AI listens continuously
3. Interrupt anytime by speaking over the AI
4. Use 🎤 to mute your input (AI keeps speaking)
5. Click ✕ to return to Chat Mode

### Settings ⚙️
- **Font Size**: Small, Medium, or Large
- **Voice Mode**: Auto-send after recording or Manual review
- **Model Selection**: Switch between available Ollama models

## Technologies Used 🛠️

- **Python & PyQt6** - Application framework and UI
- **Whisper** (via faster-whisper) - Speech-to-text
- **Ollama** (streaming mode) - Local LLM inference
- **Kokoro ONNX** - Text-to-speech synthesis
- **PyAudio/sounddevice** - Audio I/O
- **NumPy** - Audio processing

## Project Structure 📁

```
ChatbotAI-English/
├── main.py              # Main application & UI
├── ai_manager.py        # AI model coordination (Whisper, Ollama, Kokoro)
├── audio_utils.py       # Audio recording and playback
├── styles.py            # UI styling (Gemini-inspired dark theme)
├── preferences.py       # User settings persistence
├── kokoro-v0_19.onnx   # TTS model
├── voices.json          # Voice configurations
└── requirements.txt     # Python dependencies
```

## License 📄

The Unlicense
