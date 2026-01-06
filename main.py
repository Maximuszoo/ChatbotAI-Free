"""
Voice Chatbot - Google Gemini Inspired UI
Walkie-Talkie Mode: Manual control of recording
"""

import sys
import numpy as np
import threading
import queue
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QScrollArea, QPushButton, 
    QComboBox, QFrame, QSpacerItem, QSizePolicy, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QFont

from styles import GEMINI_STYLE, COLORS
from audio_utils import AudioRecorder, AudioPlayer
from ai_manager import AIManager
import ollama


def get_available_ollama_models():
    """Get list of available Ollama models"""
    try:
        response = ollama.list()
        model_names = [model.model for model in response.models]
        print(f"Found Ollama models: {model_names}")
        return model_names if model_names else ['llama3.1:8b']
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        return ['llama3.1:8b']


class ManualRecorderThread(QThread):
    """Thread for manual recording (walkie-talkie style)"""
    
    recording_finished = pyqtSignal(object)  # Emits recorded audio data
    
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.is_recording = False
        self.audio_chunks = []
    
    def run(self):
        """Record until stop() is called"""
        self.is_recording = True
        self.audio_chunks = []
        self.recorder.start_stream()
        
        print("🎤 Recording started...")
        
        while self.is_recording:
            try:
                chunk = self.recorder.audio_queue.get(timeout=0.1)
                self.audio_chunks.append(chunk)
            except:
                continue
        
        self.recorder.stop_stream()
        
        if self.audio_chunks:
            audio_data = np.concatenate(self.audio_chunks, axis=0).flatten()
            duration = len(audio_data) / self.recorder.sample_rate
            print(f"🎤 Recording stopped. Duration: {duration:.2f}s")
            
            # Only emit if we have meaningful audio (> 0.5 seconds)
            if duration > 0.5:
                self.recording_finished.emit(audio_data)
            else:
                print("Recording too short, discarding")
                self.recording_finished.emit(None)
        else:
            self.recording_finished.emit(None)
    
    def stop_recording(self):
        """Stop the recording"""
        self.is_recording = False


class WorkerThread(QThread):
    """Thread for processing: Transcribe -> Think -> Speak"""
    
    status_changed = pyqtSignal(str)
    user_message = pyqtSignal(str)
    bot_message = pyqtSignal(str)
    processing_complete = pyqtSignal()
    speaking_started = pyqtSignal()  # Signal when TTS starts playing
    
    def __init__(self, ai_manager, audio_player):
        super().__init__()
        self.ai_manager = ai_manager
        self.audio_player = audio_player
        self.audio_data = None
        self.text_input = None  # For text messages
        self.interrupted = False
    
    def set_audio(self, audio_data):
        """Set audio data to process"""
        self.audio_data = audio_data
        self.text_input = None
    
    def set_text(self, text):
        """Set text to process (skip transcription)"""
        self.text_input = text
        self.audio_data = None
    
    def interrupt(self):
        """Interrupt the current processing (stop audio)"""
        self.interrupted = True
        self.audio_player.stop()
        print("⚠️ Interrupted by user")
    
    def run(self):
        """Process the audio or text through the pipeline"""
        self.interrupted = False
        
        # Determine user text source
        if self.text_input:
            user_text = self.text_input
            self.user_message.emit(user_text)
        elif self.audio_data is not None:
            # Step 1: Transcribe
            self.status_changed.emit("Transcribing...")
            user_text = self.ai_manager.transcribe(self.audio_data)
            
            if not user_text:
                self.status_changed.emit("Ready")
                self.processing_complete.emit()
                return
            
            self.user_message.emit(user_text)
        else:
            self.processing_complete.emit()
            return
        
        if self.interrupted:
            self.processing_complete.emit()
            return
        
        # Step 2: Get LLM response
        self.status_changed.emit("Thinking...")
        bot_text = self.ai_manager.get_llm_response(user_text)
        self.bot_message.emit(bot_text)
        
        if self.interrupted:
            self.processing_complete.emit()
            return
        
        # Step 3: Generate and play speech by paragraphs with parallel processing
        self.status_changed.emit("Speaking...")
        self.speaking_started.emit()  # Notify UI that speaking started
        
        # Split text into paragraphs
        paragraphs = self._split_into_paragraphs(bot_text)
        
        # Create a queue for audio chunks
        audio_queue = queue.Queue(maxsize=3)  # Buffer up to 3 paragraphs
        generation_complete = threading.Event()
        
        # Thread function to generate audio chunks in parallel
        def generate_audio():
            for i, paragraph in enumerate(paragraphs):
                if self.interrupted:
                    break
                
                # Clean markdown from paragraph
                clean_text = self._clean_markdown(paragraph)
                
                if not clean_text.strip():
                    continue
                
                # Generate audio for this paragraph
                audio_output, sample_rate = self.ai_manager.text_to_speech(clean_text)
                
                if audio_output is not None and len(audio_output) > 0:
                    # Put audio in queue (will block if queue is full)
                    try:
                        audio_queue.put((audio_output, sample_rate), timeout=1.0)
                    except queue.Full:
                        if self.interrupted:
                            break
            
            generation_complete.set()
        
        # Start generation thread
        gen_thread = threading.Thread(target=generate_audio, daemon=True)
        gen_thread.start()
        
        # Play audio chunks as they become available
        while not (generation_complete.is_set() and audio_queue.empty()):
            if self.interrupted:
                break
            
            try:
                audio_output, sample_rate = audio_queue.get(timeout=0.5)
                if not self.interrupted:
                    self.audio_player.play(audio_output, sample_rate)
            except queue.Empty:
                if generation_complete.is_set():
                    break
                continue
        
        # Wait for generation thread to finish
        gen_thread.join(timeout=1.0)
        
        self.status_changed.emit("Ready")
        self.processing_complete.emit()
    
    def _split_into_paragraphs(self, text):
        """Split text into paragraphs for faster TTS streaming"""
        # Split by double newline or single newline
        paragraphs = text.replace('\r\n', '\n').split('\n\n')
        
        # If no double newlines, try single newlines
        if len(paragraphs) == 1:
            paragraphs = text.split('\n')
        
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _clean_markdown(self, text):
        """Remove markdown symbols like *, **, etc."""
        import re
        
        # Remove bold/italic markers
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)  # ***text***
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **text**
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # *text*
        text = re.sub(r'__(.+?)__', r'\1', text)  # __text__
        text = re.sub(r'_(.+?)_', r'\1', text)  # _text_
        
        # Remove remaining asterisks
        text = text.replace('*', '')
        text = text.replace('_', '')
        
        # Remove markdown headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`(.+?)`', r'\1', text)
        
        return text.strip()


class ChatBubble(QFrame):
    """Modern chat bubble widget"""
    
    def __init__(self, text, is_user=False):
        super().__init__()
        
        if is_user:
            self.setObjectName("userBubble")
            self.setStyleSheet("background-color: #004A77; border-radius: 20px;")
        else:
            self.setObjectName("botBubble")
            self.setStyleSheet("background-color: #1E1F20; border-radius: 20px;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(text)
        label.setObjectName("bubbleText")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setStyleSheet("""
            color: #E3E3E3;
            font-size: 15px;
            padding: 14px 18px;
            background: transparent;
        """)
        
        layout.addWidget(label)
        
        self.setMaximumWidth(600)


class MainWindow(QMainWindow):
    """Main application window - Gemini Style"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Voice Chat AI")
        self.setGeometry(100, 100, 500, 800)
        self.setMinimumSize(400, 600)
        
        # State
        self.is_recording = False
        self.is_processing = False
        self.is_speaking = False
        
        # Initialize components
        self.recorder = AudioRecorder()
        self.player = AudioPlayer()
        self.recorder_thread = None
        self.worker_thread = None
        
        # Load AI Manager
        print("Loading AI models...")
        try:
            self.ai_manager = AIManager()
        except Exception as e:
            print(f"Error initializing AI: {e}")
            self.ai_manager = None
        
        # Setup UI
        self.init_ui()
        self.setStyleSheet(GEMINI_STYLE)
        
        # Recording animation timer
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.update_recording_animation)
        self.pulse_state = 0
    
    def init_ui(self):
        """Initialize the Gemini-inspired UI"""
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ===== HEADER =====
        header = QWidget()
        header.setObjectName("headerWidget")
        header.setFixedHeight(70)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 15, 20, 15)
        
        # Model Selector (centered)
        header_layout.addStretch()
        
        self.model_selector = QComboBox()
        self.model_selector.setObjectName("modelSelector")
        available_models = get_available_ollama_models()
        self.model_selector.addItems(available_models)
        
        # Set current model
        if self.ai_manager:
            current = self.ai_manager.ollama_model
            idx = self.model_selector.findText(current)
            if idx >= 0:
                self.model_selector.setCurrentIndex(idx)
        
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        header_layout.addWidget(self.model_selector)
        
        header_layout.addStretch()
        
        main_layout.addWidget(header)
        
        # ===== CHAT AREA =====
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        self.chat_widget = QWidget()
        self.chat_widget.setObjectName("chatContainer")
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.setContentsMargins(16, 16, 16, 16)
        self.chat_layout.setSpacing(12)
        self.chat_layout.addStretch()
        
        scroll.setWidget(self.chat_widget)
        self.scroll_area = scroll
        main_layout.addWidget(scroll, 1)
        
        # ===== INPUT BAR =====
        input_bar = QWidget()
        input_bar.setObjectName("inputBar")
        input_bar.setFixedHeight(140)
        input_layout = QVBoxLayout(input_bar)
        input_layout.setContentsMargins(16, 12, 16, 16)
        input_layout.setSpacing(10)
        
        # Text input row
        text_row = QHBoxLayout()
        text_row.setSpacing(10)
        
        # Text input field
        self.text_input = QLineEdit()
        self.text_input.setObjectName("textInput")
        self.text_input.setPlaceholderText("Type a message...")
        self.text_input.setStyleSheet("""
            QLineEdit {
                background-color: #282A2C;
                color: #E3E3E3;
                border: 1px solid #3C4043;
                border-radius: 20px;
                padding: 12px 18px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #8AB4F8;
            }
        """)
        self.text_input.returnPressed.connect(self.send_text_message)
        text_row.addWidget(self.text_input)
        
        # Send button
        self.send_btn = QPushButton("➤")
        self.send_btn.setObjectName("sendButton")
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #1A73E8;
                color: white;
                border: none;
                border-radius: 20px;
                min-width: 40px;
                max-width: 40px;
                min-height: 40px;
                max-height: 40px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #4285F4;
            }
            QPushButton:disabled {
                background-color: #3C4043;
            }
        """)
        self.send_btn.clicked.connect(self.send_text_message)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        text_row.addWidget(self.send_btn)
        
        input_layout.addLayout(text_row)
        
        # Button row (mic + status)
        button_row = QHBoxLayout()
        button_row.setSpacing(15)
        
        # Clear button (left)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearButton")
        self.clear_btn.clicked.connect(self.clear_chat)
        self.clear_btn.setFixedWidth(70)
        button_row.addWidget(self.clear_btn)
        
        # Status label (center-left)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_row.addWidget(self.status_label, 1)
        
        # MIC BUTTON (center-right) - Main voice button
        self.mic_btn = QPushButton("🎤")
        self.mic_btn.setObjectName("micButton")
        self.mic_btn.clicked.connect(self.toggle_recording)
        self.mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_row.addWidget(self.mic_btn)
        
        input_layout.addLayout(button_row)
        main_layout.addWidget(input_bar)
        
        # Welcome message
        self.add_bot_message("Hello! Type a message or tap the microphone to speak.")
    
    def toggle_recording(self):
        """Toggle between recording and not recording, or interrupt if speaking"""
        
        # If speaking, interrupt
        if self.is_speaking:
            self.interrupt_speaking()
            return
        
        if self.is_processing and not self.is_speaking:
            return
        
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def interrupt_speaking(self):
        """Interrupt the bot while speaking"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.interrupt()
            # Wait with timeout to avoid blocking
            if not self.worker_thread.wait(1000):  # Wait max 1 second
                print("⚠️ Thread didn't finish in time, forcing cleanup")
            self.status_label.setText("Ready")
            self.is_speaking = False
            self.is_processing = False
            self.mic_btn.setText("🎤")
            self.mic_btn.setEnabled(True)
            self.send_btn.setEnabled(True)
            print("🛑 User interrupted playback")
    
    def send_text_message(self):
        """Send a text message typed by user"""
        text = self.text_input.text().strip()
        
        if not text:
            return
        
        if self.ai_manager is None:
            self.status_label.setText("AI not loaded!")
            return
        
        # If speaking, interrupt first
        if self.is_speaking:
            self.interrupt_speaking()
        
        if self.is_processing:
            return
        
        # Clear input
        self.text_input.clear()
        
        # Start processing
        self.is_processing = True
        self.mic_btn.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.status_label.setText("Processing...")
        
        # Start worker thread with text
        self.worker_thread = WorkerThread(self.ai_manager, self.player)
        self.worker_thread.set_text(text)
        self.worker_thread.status_changed.connect(self.update_status)
        self.worker_thread.user_message.connect(self.add_user_message)
        self.worker_thread.bot_message.connect(self.add_bot_message)
        self.worker_thread.processing_complete.connect(self.on_processing_complete)
        self.worker_thread.speaking_started.connect(self.on_speaking_started)
        self.worker_thread.start()
    
    def start_recording(self):
        """Start manual recording"""
        
        if self.ai_manager is None:
            self.status_label.setText("AI not loaded!")
            return
        
        self.is_recording = True
        
        # Update UI
        self.mic_btn.setText("⏹")
        self.mic_btn.setStyleSheet("""
            QPushButton#micButton {
                background-color: #EA4335;
                color: white;
                border: none;
                border-radius: 32px;
                min-width: 64px;
                max-width: 64px;
                min-height: 64px;
                max-height: 64px;
                font-size: 26px;
            }
            QPushButton#micButton:hover {
                background-color: #F44336;
            }
        """)
        self.status_label.setText("🔴 Recording... Tap to send")
        
        # Start recording thread
        self.recorder_thread = ManualRecorderThread(self.recorder)
        self.recorder_thread.recording_finished.connect(self.on_recording_finished)
        self.recorder_thread.start()
        
        # Start pulse animation
        self.pulse_timer.start(500)
    
    def stop_recording(self):
        """Stop recording and process"""
        
        self.is_recording = False
        self.pulse_timer.stop()
        
        # Update UI
        self.mic_btn.setText("🎤")
        self.mic_btn.setStyleSheet("""
            QPushButton#micButton {
                background-color: #1A73E8;
                color: white;
                border: none;
                border-radius: 32px;
                min-width: 64px;
                max-width: 64px;
                min-height: 64px;
                max-height: 64px;
                font-size: 26px;
            }
            QPushButton#micButton:hover {
                background-color: #4285F4;
            }
        """)
        
        # Stop the recorder thread
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop_recording()
    
    def update_recording_animation(self):
        """Pulse animation for recording state"""
        self.pulse_state = (self.pulse_state + 1) % 2
        if self.pulse_state == 0:
            self.status_label.setText("🔴 Recording... Tap to send")
        else:
            self.status_label.setText("⚫ Recording... Tap to send")
    
    @pyqtSlot(object)
    def on_recording_finished(self, audio_data):
        """Handle finished recording"""
        
        if audio_data is None:
            self.status_label.setText("Ready")
            return
        
        self.is_processing = True
        self.mic_btn.setEnabled(False)
        self.status_label.setText("Processing...")
        
        # Start worker thread
        self.worker_thread = WorkerThread(self.ai_manager, self.player)
        self.worker_thread.set_audio(audio_data)
        self.worker_thread.status_changed.connect(self.update_status)
        self.worker_thread.user_message.connect(self.add_user_message)
        self.worker_thread.bot_message.connect(self.add_bot_message)
        self.worker_thread.processing_complete.connect(self.on_processing_complete)
        self.worker_thread.speaking_started.connect(self.on_speaking_started)
        self.worker_thread.start()
    
    @pyqtSlot()
    def on_processing_complete(self):
        """Called when processing is done"""
        self.is_processing = False
        self.is_speaking = False
        self.mic_btn.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.mic_btn.setText("🎤")
        self.status_label.setText("Ready")
    
    @pyqtSlot()
    def on_speaking_started(self):
        """Called when TTS starts playing audio"""
        self.is_speaking = True
        self.mic_btn.setEnabled(True)  # Enable to allow interruption
        self.mic_btn.setText("⏹️")  # Change to stop icon
    
    @pyqtSlot(str)
    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(status)
    
    @pyqtSlot(str)
    def on_model_changed(self, model_name):
        """Handle model selection change"""
        if self.ai_manager:
            self.ai_manager.ollama_model = model_name
            print(f"Switched to: {model_name}")
            self.add_bot_message(f"[Model: {model_name}]")
    
    @pyqtSlot(str)
    def add_user_message(self, message):
        """Add user bubble"""
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(60, 0, 12, 0)
        wrapper_layout.addStretch()
        
        bubble = ChatBubble(message, is_user=True)
        wrapper_layout.addWidget(bubble)
        
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, wrapper)
        self.scroll_to_bottom()
    
    @pyqtSlot(str)
    def add_bot_message(self, message):
        """Add bot bubble"""
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(12, 0, 60, 0)
        
        bubble = ChatBubble(message, is_user=False)
        wrapper_layout.addWidget(bubble)
        wrapper_layout.addStretch()
        
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, wrapper)
        self.scroll_to_bottom()
    
    def scroll_to_bottom(self):
        """Scroll to bottom of chat"""
        QTimer.singleShot(100, lambda: 
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
        )
    
    def clear_chat(self):
        """Clear all messages"""
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if self.ai_manager:
            self.ai_manager.reset_conversation()
        
        self.add_bot_message("Chat cleared. Tap the mic to start a new conversation!")
    
    def closeEvent(self, event):
        """Cleanup on close"""
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop_recording()
            self.recorder_thread.wait()
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.wait()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    font = QFont("Google Sans", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
