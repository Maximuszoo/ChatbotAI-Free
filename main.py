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
    QComboBox, QFrame, QSpacerItem, QSizePolicy, QLineEdit,
    QDialog, QCheckBox, QStackedWidget, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import QFont, QColor

from styles import GEMINI_STYLE, COLORS
from audio_utils import AudioRecorder, AudioPlayer
from ai_manager import AIManager
from preferences import load_preferences, save_preferences, get_font_size_config, FONT_SIZES
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
    """Thread for processing: Transcribe -> Think -> Speak with real-time streaming"""
    
    status_changed = pyqtSignal(str)
    user_message = pyqtSignal(str)
    bot_message = pyqtSignal(str)
    bot_message_update = pyqtSignal(str)  # For streaming text updates
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
        """Process the audio or text through the pipeline with streaming"""
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
        
        # Step 2 & 3: Stream LLM response and generate TTS in parallel
        self.status_changed.emit("Thinking...")
        
        # Queue for sentences ready for TTS
        sentence_queue = queue.Queue()
        # Queue for audio ready to play
        audio_queue = queue.Queue(maxsize=3)
        
        llm_complete = threading.Event()
        tts_complete = threading.Event()
        first_audio_ready = threading.Event()
        
        # Track the full response for history
        full_response = [""]
        bot_bubble_created = [False]
        
        def on_text_chunk(text_so_far):
            """Called for each token - update UI in real-time"""
            if self.interrupted:
                return
            full_response[0] = text_so_far
            
            # Create or update the bot bubble
            if not bot_bubble_created[0]:
                self.bot_message.emit(text_so_far)
                bot_bubble_created[0] = True
            else:
                self.bot_message_update.emit(text_so_far)
        
        def on_sentence_ready(sentence):
            """Called when a complete sentence is ready for TTS"""
            if self.interrupted:
                return
            clean_sentence = self._clean_markdown(sentence)
            if clean_sentence.strip():
                sentence_queue.put(clean_sentence)
        
        # Thread 1: Stream LLM response
        def stream_llm():
            try:
                self.ai_manager.get_llm_response_streaming(
                    user_text,
                    on_chunk=on_text_chunk,
                    on_sentence=on_sentence_ready
                )
            except Exception as e:
                print(f"LLM streaming error: {e}")
            finally:
                llm_complete.set()
        
        # Thread 2: Generate TTS for sentences as they arrive
        def generate_tts():
            while not (llm_complete.is_set() and sentence_queue.empty()):
                if self.interrupted:
                    break
                
                try:
                    sentence = sentence_queue.get(timeout=0.1)
                    if sentence and not self.interrupted:
                        audio_output, sample_rate = self.ai_manager.text_to_speech(sentence)
                        if audio_output is not None and len(audio_output) > 0:
                            audio_queue.put((audio_output, sample_rate))
                            if not first_audio_ready.is_set():
                                first_audio_ready.set()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"TTS error: {e}")
            
            tts_complete.set()
        
        # Start LLM and TTS threads
        llm_thread = threading.Thread(target=stream_llm, daemon=True)
        tts_thread = threading.Thread(target=generate_tts, daemon=True)
        
        llm_thread.start()
        tts_thread.start()
        
        # Wait for first audio to be ready, then start playing
        first_audio_ready.wait(timeout=30)  # Max 30 seconds for first response
        
        if not self.interrupted:
            self.status_changed.emit("Speaking...")
            self.speaking_started.emit()
        
        # Play audio chunks as they become available
        while not (tts_complete.is_set() and audio_queue.empty()):
            if self.interrupted:
                break
            
            try:
                audio_output, sample_rate = audio_queue.get(timeout=0.3)
                if not self.interrupted:
                    self.audio_player.play(audio_output, sample_rate)
            except queue.Empty:
                if tts_complete.is_set():
                    break
                continue
        
        # Wait for threads to finish
        llm_thread.join(timeout=1.0)
        tts_thread.join(timeout=1.0)
        
        self.status_changed.emit("Ready")
        self.processing_complete.emit()
    
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
    """Modern chat bubble widget - responsive to window size"""
    
    def __init__(self, text, is_user=False, font_size=15):
        super().__init__()
        
        if is_user:
            self.setObjectName("userBubble")
            self.setStyleSheet("background-color: #004A77; border-radius: 20px;")
        else:
            self.setObjectName("botBubble")
            self.setStyleSheet("background-color: #1E1F20; border-radius: 20px;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(text)
        self.label.setObjectName("bubbleText")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.update_font_size(font_size)
        
        layout.addWidget(self.label)
        
        # Responsive: use size policy instead of fixed max width
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    
    def update_text(self, text):
        """Update the bubble text content (for streaming)"""
        self.label.setText(text)
    
    def update_font_size(self, font_size):
        """Update the font size of the bubble text"""
        self.label.setStyleSheet(f"""
            color: #E3E3E3;
            font-size: {font_size}px;
            padding: 14px 18px;
            background: transparent;
        """)


class LiveWorkerThread(QThread):
    """Thread for Live Mode: Continuous conversation with VAD and barge-in detection"""
    
    status_changed = pyqtSignal(str)  # listening, thinking, speaking, muted
    transcription_ready = pyqtSignal(str)
    response_chunk = pyqtSignal(str)  # Streaming response
    speaking_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, ai_manager, audio_recorder, audio_player):
        super().__init__()
        self.ai_manager = ai_manager
        self.recorder = audio_recorder
        self.player = audio_player
        self.is_running = False
        self.is_muted = False
        self.interrupted = False
        self.user_speaking = threading.Event()  # Signal for barge-in
        self.monitor_thread = None
    
    def run(self):
        """Main loop for continuous conversation with barge-in detection"""
        self.is_running = True
        self.recorder.start_stream()
        
        # Start continuous audio monitoring thread for barge-in detection
        self.monitor_thread = threading.Thread(target=self._monitor_for_barge_in, daemon=True)
        self.monitor_thread.start()
        
        print("🎙️ Live Mode started with barge-in detection")
        
        while self.is_running:
            # Cuando está muted, no procesamos nueva entrada del usuario
            # pero permitimos que la IA termine de hablar si ya empezó
            if self.is_muted:
                self.status_changed.emit("muted")
                QThread.msleep(200)
                continue
            
            # Step 1: Listen for speech
            self.status_changed.emit("listening")
            
            # Record until silence
            audio_data = self._record_with_vad()
            
            if audio_data is None or not self.is_running:
                continue
            
            if self.is_muted:
                continue
            
            # Step 2: Transcribe
            self.status_changed.emit("thinking")
            user_text = self.ai_manager.transcribe(audio_data)
            
            if not user_text or not self.is_running:
                continue
            
            # No mostrar transcripción en UI
            # self.transcription_ready.emit(user_text)
            
            # Step 3: Get response with streaming
            self.interrupted = False
            self.user_speaking.clear()  # Reset barge-in flag
            full_response = ""
            
            # Queue for sentences ready for TTS
            sentence_queue = queue.Queue()
            audio_queue = queue.Queue(maxsize=3)
            llm_complete = threading.Event()
            tts_complete = threading.Event()
            
            def on_text_chunk(text_so_far):
                if self.interrupted or not self.is_running:
                    return
                self.response_chunk.emit(text_so_far)
            
            def on_sentence_ready(sentence):
                if self.interrupted or not self.is_running:
                    return
                clean_sentence = self._clean_markdown(sentence)
                if clean_sentence.strip():
                    sentence_queue.put(clean_sentence)
            
            # Thread for LLM streaming
            def stream_llm():
                nonlocal full_response
                try:
                    full_response = self.ai_manager.get_llm_response_streaming(
                        user_text,
                        on_chunk=on_text_chunk,
                        on_sentence=on_sentence_ready
                    )
                except Exception as e:
                    print(f"LLM error: {e}")
                finally:
                    llm_complete.set()
            
            # Thread for TTS generation
            def generate_tts():
                while not (llm_complete.is_set() and sentence_queue.empty()):
                    if self.interrupted or not self.is_running:
                        break
                    try:
                        sentence = sentence_queue.get(timeout=0.1)
                        if sentence:
                            audio_output, sample_rate = self.ai_manager.text_to_speech(sentence)
                            if audio_output is not None and len(audio_output) > 0:
                                audio_queue.put((audio_output, sample_rate))
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"TTS error: {e}")
                tts_complete.set()
            
            # Start threads
            llm_thread = threading.Thread(target=stream_llm, daemon=True)
            tts_thread = threading.Thread(target=generate_tts, daemon=True)
            llm_thread.start()
            tts_thread.start()
            
            # Play audio (keep monitoring for user interruption)
            self.status_changed.emit("speaking")
            # NO pausar el micrófono para poder detectar interrupciones
            
            interruption_detected = False
            
            while not (tts_complete.is_set() and audio_queue.empty()):
                # Check for user interruption (barge-in) FIRST
                if self.user_speaking.is_set():
                    if not interruption_detected:
                        print("⚠️ User interrupted! Stopping playback immediately...")
                        interruption_detected = True
                        self.interrupted = True
                        self.player.stop()
                        # Clear queues
                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        break
                
                if self.interrupted or not self.is_running:
                    break
                    
                try:
                    audio_output, sample_rate = audio_queue.get(timeout=0.2)
                    # Verificar de nuevo antes de reproducir
                    if self.user_speaking.is_set():
                        print("⚠️ Interruption detected before playback")
                        break
                    if not self.interrupted and self.is_running:
                        self.player.play(audio_output, sample_rate)
                except queue.Empty:
                    if tts_complete.is_set():
                        break
                    continue
            
            llm_thread.join(timeout=1.0)
            tts_thread.join(timeout=1.0)
            
            self.speaking_finished.emit()
            
            # If user interrupted, restart immediately to listen
            if self.user_speaking.is_set():
                print("♻️ User interrupted, restarting listening immediately")
                self.user_speaking.clear()
                continue
            
            # Small delay before listening again
            if self.is_running and not self.is_muted:
                QThread.msleep(300)
        
        self.recorder.stop_stream()
        print("🎙️ Live Mode stopped")
    
    def _monitor_for_barge_in(self):
        """Continuously monitor audio for user speech (barge-in detection)"""
        print("🎧 Starting barge-in monitoring thread")
        speech_threshold = self.recorder.silence_threshold * 2.0  # Higher threshold for clear interruption
        consecutive_speech_frames = 0
        frames_needed = 4  # Need 4 consecutive frames to trigger barge-in (more reliable)
        
        while self.is_running:
            # No verificar is_muted aquí - siempre monitorear para detectar interrupciones
            try:
                # Revisar directamente la cola de audio
                if not self.recorder.audio_queue.empty():
                    try:
                        # Peek sin remover
                        items = list(self.recorder.audio_queue.queue)
                        if items:
                            audio_chunk = items[-1]  # Tomar el más reciente
                            
                            rms = np.sqrt(np.mean(audio_chunk ** 2))
                            
                            if rms > speech_threshold:
                                consecutive_speech_frames += 1
                                if consecutive_speech_frames >= frames_needed:
                                    # User is speaking! Signal barge-in
                                    if not self.user_speaking.is_set():
                                        print(f"🗣️ User speech detected! RMS: {rms:.4f} (threshold: {speech_threshold:.4f})")
                                        self.user_speaking.set()
                            else:
                                # Reset if below threshold
                                if consecutive_speech_frames > 0:
                                    consecutive_speech_frames = max(0, consecutive_speech_frames - 1)
                    except (IndexError, AttributeError):
                        pass
                            
                QThread.msleep(30)  # Check every 30ms for faster response
            except Exception as e:
                QThread.msleep(100)
                continue
        
        print("🎧 Barge-in monitoring stopped")
    
    def _record_with_vad(self):
        """Record audio with Voice Activity Detection"""
        audio_buffer = []
        silence_frames = 0
        silence_threshold = self.recorder.silence_threshold
        silence_frames_needed = int(1.5 * self.recorder.sample_rate / 1024)  # 1.5 sec silence
        max_frames = int(30 * self.recorder.sample_rate / 1024)  # 30 sec max
        frame_count = 0
        speech_detected = False
        
        # Clear queue
        while not self.recorder.audio_queue.empty():
            try:
                self.recorder.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        while frame_count < max_frames and self.is_running and not self.is_muted:
            try:
                audio_chunk = self.recorder.audio_queue.get(timeout=0.1)
                audio_buffer.append(audio_chunk)
                frame_count += 1
                
                rms = np.sqrt(np.mean(audio_chunk ** 2))
                
                if rms > silence_threshold:
                    speech_detected = True
                    silence_frames = 0
                else:
                    if speech_detected:
                        silence_frames += 1
                        if silence_frames >= silence_frames_needed:
                            break
            except queue.Empty:
                continue
        
        if not audio_buffer or not speech_detected:
            return None
        
        audio_data = np.concatenate(audio_buffer, axis=0).flatten()
        duration = len(audio_data) / self.recorder.sample_rate
        
        if duration < 0.5:
            return None
        
        return audio_data
    
    def _clean_markdown(self, text):
        """Remove markdown symbols"""
        import re
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = text.replace('*', '').replace('_', '')
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`(.+?)`', r'\1', text)
        return text.strip()
    
    def set_muted(self, muted):
        """Toggle mute state - only affects user input capture, not AI playback"""
        self.is_muted = muted
        if muted:
            print("🔇 Muted - User input disabled (AI continues)")
        else:
            print("🎙️ Unmuted - User input enabled")
    
    def interrupt(self):
        """Interrupt current response"""
        self.interrupted = True
        self.player.stop()
    
    def stop(self):
        """Stop the live mode"""
        self.is_running = False
        self.interrupted = True
        self.player.stop()


class LiveModeWidget(QWidget):
    """Gemini Live-style continuous conversation interface"""
    
    exit_requested = pyqtSignal()
    
    def __init__(self, ai_manager, audio_recorder, audio_player, parent=None):
        super().__init__(parent)
        self.ai_manager = ai_manager
        self.recorder = audio_recorder
        self.player = audio_player
        self.worker_thread = None
        self.is_active = False
        
        self.setObjectName("liveModeWidget")
        self.init_ui()
        self.setup_animations()
    
    def init_ui(self):
        """Create the Live Mode UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 40, 20, 30)
        layout.setSpacing(20)
        
        # Top spacer
        layout.addStretch(1)
        
        # Central indicator container
        indicator_container = QWidget()
        indicator_container.setObjectName("liveIndicatorContainer")
        indicator_layout = QVBoxLayout(indicator_container)
        indicator_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        indicator_layout.setSpacing(20)
        
        # Pulsing circle indicator
        self.indicator = QLabel()
        self.indicator.setFixedSize(150, 150)
        self.indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.indicator.setStyleSheet("""
            background-color: #1A73E8;
            border-radius: 75px;
        """)
        
        # Add glow effect
        self.glow_effect = QGraphicsDropShadowEffect()
        self.glow_effect.setBlurRadius(40)
        self.glow_effect.setColor(QColor(138, 180, 248, 150))
        self.glow_effect.setOffset(0, 0)
        self.indicator.setGraphicsEffect(self.glow_effect)
        
        indicator_layout.addWidget(self.indicator, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # State label
        self.state_label = QLabel("Ready to start")
        self.state_label.setObjectName("liveStateLabel")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        indicator_layout.addWidget(self.state_label)
        
        layout.addWidget(indicator_container, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addStretch(2)
        
        # Bottom controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setSpacing(40)
        controls_layout.setContentsMargins(0, 0, 0, 20)
        
        # Mute button
        self.mute_btn = QPushButton("🎤")
        self.mute_btn.setObjectName("liveMuteButton")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setChecked(False)
        self.mute_btn.clicked.connect(self.toggle_mute)
        self.mute_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        controls_layout.addWidget(self.mute_btn)
        
        # Exit button
        self.exit_btn = QPushButton("✕")
        self.exit_btn.setObjectName("liveExitButton")
        self.exit_btn.clicked.connect(self.exit_live_mode)
        self.exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        controls_layout.addWidget(self.exit_btn)
        
        layout.addWidget(controls, alignment=Qt.AlignmentFlag.AlignCenter)
    
    def setup_animations(self):
        """Setup pulse animation for indicator"""
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.animate_pulse)
        self.pulse_phase = 0
    
    def animate_pulse(self):
        """Animate the indicator pulse"""
        self.pulse_phase = (self.pulse_phase + 1) % 20
        
        # Vary blur radius for pulsing effect
        base_blur = 30
        pulse_amount = 20 * (0.5 + 0.5 * np.sin(self.pulse_phase * np.pi / 10))
        self.glow_effect.setBlurRadius(base_blur + pulse_amount)
    
    def start_live_mode(self):
        """Start the live conversation mode"""
        if self.worker_thread and self.worker_thread.isRunning():
            return
        
        self.is_active = True
        self.update_state("listening")
        
        # Start worker thread
        self.worker_thread = LiveWorkerThread(
            self.ai_manager, self.recorder, self.player
        )
        self.worker_thread.status_changed.connect(self.update_state)
        # No conectar transcription ni response_chunk para no mostrar texto
        # self.worker_thread.transcription_ready.connect(self.on_transcription)
        # self.worker_thread.response_chunk.connect(self.on_response_chunk)
        self.worker_thread.speaking_finished.connect(self.on_speaking_finished)
        self.worker_thread.start()
        
        self.pulse_timer.start(50)
    
    def stop_live_mode(self):
        """Stop the live conversation mode"""
        self.is_active = False
        self.pulse_timer.stop()
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait(2000)
        
        self.update_state("stopped")
        self.state_label.setText("Ready to start")
    
    @pyqtSlot(str)
    def update_state(self, state):
        """Update visual state of the indicator"""
        if state == "listening":
            self.indicator.setStyleSheet("""
                background-color: #1A73E8;
                border-radius: 75px;
            """)
            self.glow_effect.setColor(QColor(26, 115, 232, 180))
            self.state_label.setText("Listening...")
            
        elif state == "thinking":
            self.indicator.setStyleSheet("""
                background-color: #9334E6;
                border-radius: 75px;
            """)
            self.glow_effect.setColor(QColor(147, 52, 230, 180))
            self.state_label.setText("Thinking...")
            
        elif state == "speaking":
            self.indicator.setStyleSheet("""
                background-color: #E8EAED;
                border-radius: 75px;
            """)
            self.glow_effect.setColor(QColor(232, 234, 237, 150))
            self.state_label.setText("Speaking...")
            
        elif state == "muted":
            self.indicator.setStyleSheet("""
                background-color: #5F6368;
                border-radius: 75px;
            """)
            self.glow_effect.setColor(QColor(95, 99, 104, 100))
            self.state_label.setText("Muted")
            
        elif state == "stopped":
            self.indicator.setStyleSheet("""
                background-color: #3C4043;
                border-radius: 75px;
            """)
            self.glow_effect.setColor(QColor(60, 64, 67, 80))
    
    @pyqtSlot()
    def on_speaking_finished(self):
        """Called when TTS finishes"""
        pass  # No need to update text
    
    def toggle_mute(self):
        """Toggle mute state"""
        is_muted = self.mute_btn.isChecked()
        self.mute_btn.setText("🔇" if is_muted else "🎤")
        
        if self.worker_thread:
            self.worker_thread.set_muted(is_muted)
    
    def exit_live_mode(self):
        """Exit live mode and return to chat"""
        self.stop_live_mode()
        self.mute_btn.setChecked(False)
        self.mute_btn.setText("🎤")
        self.exit_requested.emit()


class SettingsDialog(QDialog):
    """Settings dialog with font size options"""
    
    def __init__(self, parent=None, auto_send=True, font_size="medium"):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(350)
        self.auto_send = auto_send
        self.font_size = font_size
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("⚙️ Settings")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #E3E3E3;")
        layout.addWidget(title)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #3C4043;")
        layout.addWidget(separator)
        
        # Font size setting
        font_label = QLabel("Font Size:")
        font_label.setStyleSheet("font-size: 14px; color: #E3E3E3; margin-top: 10px;")
        layout.addWidget(font_label)
        
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems(["Small", "Medium", "Large"])
        size_map = {"small": 0, "medium": 1, "large": 2}
        self.font_size_combo.setCurrentIndex(size_map.get(font_size, 1))
        self.font_size_combo.setStyleSheet("""
            QComboBox {
                background-color: #282A2C;
                color: #E3E3E3;
                border: 1px solid #3C4043;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                min-width: 120px;
            }
            QComboBox:hover {
                background-color: #3C4043;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #9AA0A6;
            }
            QComboBox QAbstractItemView {
                background-color: #282A2C;
                color: #E3E3E3;
                selection-background-color: #3C4043;
                border: 1px solid #3C4043;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        layout.addWidget(self.font_size_combo)
        
        # Separator 2
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet("background-color: #3C4043;")
        layout.addWidget(separator2)
        
        # Voice mode setting
        mode_label = QLabel("Voice Recording Mode:")
        mode_label.setStyleSheet("font-size: 14px; color: #E3E3E3; margin-top: 10px;")
        layout.addWidget(mode_label)
        
        self.auto_send_checkbox = QCheckBox("Send automatically after recording")
        self.auto_send_checkbox.setChecked(auto_send)
        self.auto_send_checkbox.setStyleSheet("""
            QCheckBox {
                color: #E3E3E3;
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #5F6368;
                background-color: #202124;
            }
            QCheckBox::indicator:checked {
                background-color: #1A73E8;
                border-color: #1A73E8;
            }
        """)
        layout.addWidget(self.auto_send_checkbox)
        
        help_text = QLabel("When unchecked, transcribed text will appear in the input field for you to review before sending.")
        help_text.setWordWrap(True)
        help_text.setStyleSheet("font-size: 11px; color: #9AA0A6; margin-left: 26px;")
        layout.addWidget(help_text)
        
        layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Save & Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #1A73E8;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1557b0;
            }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        # Set dialog background
        self.setStyleSheet("""
            QDialog {
                background-color: #202124;
            }
        """)
    
    def get_auto_send(self):
        return self.auto_send_checkbox.isChecked()
    
    def get_font_size(self):
        size_names = ["small", "medium", "large"]
        return size_names[self.font_size_combo.currentIndex()]


class MainWindow(QMainWindow):
    """Main application window - Gemini Style"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Voice Chat AI")
        self.setGeometry(100, 100, 500, 800)
        self.setMinimumSize(400, 600)
        
        # Load user preferences
        self.preferences = load_preferences()
        
        # State
        self.is_recording = False
        self.is_processing = False
        self.is_speaking = False
        self.auto_send = self.preferences.get("auto_send", True)
        self.font_size_name = self.preferences.get("font_size", "medium")
        self.chat_bubbles = []  # Track bubbles for font size updates
        
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
        self.apply_font_size()  # Apply saved font size
        
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
        
        # Set model from preferences or use first available
        saved_model = self.preferences.get("model")
        if saved_model and saved_model in available_models:
            idx = self.model_selector.findText(saved_model)
            if idx >= 0:
                self.model_selector.setCurrentIndex(idx)
                if self.ai_manager:
                    self.ai_manager.ollama_model = saved_model
        elif available_models:
            # Use first available model as default
            self.model_selector.setCurrentIndex(0)
            if self.ai_manager:
                self.ai_manager.ollama_model = available_models[0]
        
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        header_layout.addWidget(self.model_selector)
        
        header_layout.addStretch()
        
        # Settings button (top right)
        settings_btn = QPushButton("⚙️")
        settings_btn.setObjectName("settingsButton")
        settings_btn.setFixedSize(40, 40)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 20px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #3C4043;
            }
        """)
        settings_btn.clicked.connect(self.open_settings)
        header_layout.addWidget(settings_btn)
        
        main_layout.addWidget(header)
        
        # ===== STACKED WIDGET (Chat + Live Mode) =====
        self.stacked_widget = QStackedWidget()
        
        # --- Page 0: Chat Area ---
        chat_page = QWidget()
        chat_page_layout = QVBoxLayout(chat_page)
        chat_page_layout.setContentsMargins(0, 0, 0, 0)
        chat_page_layout.setSpacing(0)
        
        # Chat scroll area
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
        chat_page_layout.addWidget(scroll, 1)
        
        self.stacked_widget.addWidget(chat_page)  # Page 0: Chat
        
        # --- Page 1: Live Mode ---
        self.live_mode_widget = LiveModeWidget(
            self.ai_manager, self.recorder, self.player, self
        )
        self.live_mode_widget.exit_requested.connect(self.exit_live_mode)
        self.stacked_widget.addWidget(self.live_mode_widget)  # Page 1: Live
        
        main_layout.addWidget(self.stacked_widget, 1)
        
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
        
        # LIVE MODE BUTTON - Start continuous conversation
        self.live_btn = QPushButton("✨")
        self.live_btn.setObjectName("liveStartButton")
        self.live_btn.setToolTip("Start Live Mode")
        self.live_btn.clicked.connect(self.enter_live_mode)
        self.live_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_row.addWidget(self.live_btn)
        
        input_layout.addLayout(button_row)
        main_layout.addWidget(input_bar)
        
        # Store reference to input_bar for showing/hiding
        self.input_bar = input_bar
        
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
        self.worker_thread.bot_message_update.connect(self.update_bot_message)
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
        
        # If auto_send is disabled, transcribe and put in text field
        if not self.auto_send:
            self.status_label.setText("Transcribing...")
            self.mic_btn.setEnabled(False)
            self.send_btn.setEnabled(False)
            
            # Transcribe in a background thread
            def transcribe_only():
                text = self.ai_manager.transcribe(audio_data)
                return text
            
            # Use QTimer to run transcription without blocking
            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(transcribe_only)
            
            def on_transcribe_done():
                try:
                    transcribed_text = future.result(timeout=0.1)
                    if transcribed_text:
                        self.text_input.setText(transcribed_text)
                        self.text_input.setFocus()
                        self.status_label.setText("Review and send")
                    else:
                        self.status_label.setText("Ready")
                except concurrent.futures.TimeoutError:
                    # Not ready yet, check again
                    QTimer.singleShot(100, on_transcribe_done)
                    return
                except Exception as e:
                    print(f"Transcription error: {e}")
                    self.status_label.setText("Ready")
                finally:
                    if future.done():
                        self.mic_btn.setEnabled(True)
                        self.send_btn.setEnabled(True)
                        executor.shutdown(wait=False)
            
            QTimer.singleShot(100, on_transcribe_done)
            return
        
        # Auto-send mode (original behavior)
        self.is_processing = True
        self.mic_btn.setEnabled(False)
        self.status_label.setText("Processing...")
        
        # Start worker thread
        self.worker_thread = WorkerThread(self.ai_manager, self.player)
        self.worker_thread.set_audio(audio_data)
        self.worker_thread.status_changed.connect(self.update_status)
        self.worker_thread.user_message.connect(self.add_user_message)
        self.worker_thread.bot_message.connect(self.add_bot_message)
        self.worker_thread.bot_message_update.connect(self.update_bot_message)
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
    
    def enter_live_mode(self):
        """Switch to Live Mode (continuous conversation)"""
        if self.ai_manager is None:
            self.status_label.setText("AI not loaded!")
            return
        
        # Stop any ongoing recording/processing
        if self.is_recording:
            self.stop_recording()
        if self.is_speaking:
            self.interrupt_speaking()
        
        # Hide input bar when entering live mode
        self.input_bar.setVisible(False)
        
        # Switch to Live Mode page
        self.stacked_widget.setCurrentIndex(1)
        self.live_mode_widget.start_live_mode()
        print("🎙️ Entering Live Mode")
    
    def exit_live_mode(self):
        """Return to Chat Mode from Live Mode"""
        # Show input bar again
        self.input_bar.setVisible(True)
        
        self.stacked_widget.setCurrentIndex(0)
        self.status_label.setText("Ready")
        print("💬 Returning to Chat Mode")
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self, self.auto_send, self.font_size_name)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.auto_send = dialog.get_auto_send()
            new_font_size = dialog.get_font_size()
            
            # Update font size if changed
            if new_font_size != self.font_size_name:
                self.font_size_name = new_font_size
                self.apply_font_size()
            
            # Save preferences
            self.preferences["auto_send"] = self.auto_send
            self.preferences["font_size"] = self.font_size_name
            save_preferences(self.preferences)
            
            mode_name = "Auto-send" if self.auto_send else "Manual review"
            print(f"⚙️ Settings updated - Mode: {mode_name}, Font: {self.font_size_name}")
    
    def apply_font_size(self):
        """Apply font size to all UI elements"""
        font_config = get_font_size_config(self.font_size_name)
        
        # Update existing bubbles
        for bubble in self.chat_bubbles:
            if bubble:
                try:
                    bubble.update_font_size(font_config["bubble_text"])
                except RuntimeError:
                    pass  # Widget was deleted
        
        # Update input field
        self.text_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: #282A2C;
                color: #E3E3E3;
                border: 1px solid #3C4043;
                border-radius: 20px;
                padding: 12px 18px;
                font-size: {font_config["input_text"]}px;
            }}
            QLineEdit:focus {{
                border-color: #8AB4F8;
            }}
        """)
        
        # Update status label
        self.status_label.setStyleSheet(f"""
            color: #9AA0A6;
            font-size: {font_config["status_text"]}px;
            padding: 8px 16px;
            background: transparent;
        """)
    
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
            # Save preference
            self.preferences["model"] = model_name
            save_preferences(self.preferences)
    
    @pyqtSlot(str)
    def add_user_message(self, message):
        """Add user bubble"""
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        # Use percentage margins for responsive design
        wrapper_layout.setContentsMargins(40, 0, 12, 0)
        wrapper_layout.addStretch()
        
        font_config = get_font_size_config(self.font_size_name)
        bubble = ChatBubble(message, is_user=True, font_size=font_config["bubble_text"])
        self.chat_bubbles.append(bubble)
        wrapper_layout.addWidget(bubble)
        
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, wrapper)
        self.scroll_to_bottom()
    
    @pyqtSlot(str)
    def add_bot_message(self, message):
        """Add bot bubble"""
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        # Use percentage margins for responsive design
        wrapper_layout.setContentsMargins(12, 0, 40, 0)
        
        font_config = get_font_size_config(self.font_size_name)
        bubble = ChatBubble(message, is_user=False, font_size=font_config["bubble_text"])
        self.chat_bubbles.append(bubble)
        self.current_bot_bubble = bubble  # Track for streaming updates
        wrapper_layout.addWidget(bubble)
        wrapper_layout.addStretch()
        
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, wrapper)
        self.scroll_to_bottom()
    
    @pyqtSlot(str)
    def update_bot_message(self, message):
        """Update the current bot bubble text (for streaming)"""
        if hasattr(self, 'current_bot_bubble') and self.current_bot_bubble:
            try:
                self.current_bot_bubble.update_text(message)
                self.scroll_to_bottom()
            except RuntimeError:
                pass  # Widget was deleted
    
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
        
        # Clear bubbles list
        self.chat_bubbles.clear()
        
        if self.ai_manager:
            self.ai_manager.reset_conversation()
        
        self.add_bot_message("Chat cleared. Tap the mic to start a new conversation!")
    
    def closeEvent(self, event):
        """Cleanup on close"""
        # Stop Live Mode if active
        if hasattr(self, 'live_mode_widget') and self.live_mode_widget.is_active:
            self.live_mode_widget.stop_live_mode()
        
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
