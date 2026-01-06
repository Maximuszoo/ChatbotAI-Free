"""
AI Manager - Handles Whisper STT, Ollama LLM, and Kokoro TTS
Loads models once at initialization to save VRAM
"""

import os
import json
import numpy as np
from faster_whisper import WhisperModel
import ollama
try:
    from kokoro_wrapper import KokoroWrapper as Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("Warning: kokoro-onnx not installed, will use fallback TTS")


class AIManager:
    """Manages all AI models for the chatbot"""
    
    def __init__(self, 
                 whisper_model="base.en",
                 ollama_model="llama3.1:8b",
                 kokoro_model_path="kokoro-v0_19.onnx",
                 voices_path="voices.json",
                 voice_name="af_bella"):
        """
        Initialize all AI models
        
        Args:
            whisper_model: faster-whisper model size
            ollama_model: Ollama model name
            kokoro_model_path: Path to Kokoro ONNX model
            voices_path: Path to voices.json
            voice_name: Voice to use (af_bella or af_sarah)
        """
        print("Initializing AI Manager...")
        
        # Check CUDA availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {cuda_available}")
            device = "cuda" if cuda_available else "cpu"
            compute_type = "float16" if cuda_available else "int8"
        except ImportError:
            print("PyTorch not found, assuming CUDA available")
            device = "cuda"
            compute_type = "float16"
        
        # Load Whisper STT
        print(f"Loading Whisper model: {whisper_model} on {device}")
        print("This may take a few minutes on first run (downloading model)...")
        self.whisper = WhisperModel(
            whisper_model,
            device=device,
            compute_type=compute_type
        )
        print("✓ Whisper model loaded successfully!")
        
        # Ollama LLM
        print(f"✓ Ollama configured with model: {ollama_model}")
        self.ollama_model = ollama_model
        
        # Load Kokoro TTS
        print(f"Loading Kokoro TTS model: {kokoro_model_path}")
        print("Setting up ONNX runtime...")
        
        if KOKORO_AVAILABLE:
            try:
                # Kokoro needs the model file and voices file
                print(f"Initializing Kokoro with voice: {voice_name}")
                self.kokoro = Kokoro(kokoro_model_path, voices_path)
                self.voice_name = voice_name
                
                # Test if the voice works
                print("Testing TTS with a sample phrase...")
                test_audio, test_sr = self.kokoro.create("Hello", voice=voice_name)
                print(f"✓ Kokoro TTS model loaded successfully!")
                print(f"✓ Voice loaded: {voice_name}")
                print(f"✓ Test generated {len(test_audio)} samples at {test_sr}Hz")
                self.tts_available = True
            except Exception as e:
                print(f"Warning: Could not load Kokoro: {e}")
                print("TTS will use fallback method")
                import traceback
                traceback.print_exc()
                self.tts_available = False
        else:
            print("Kokoro not available, TTS disabled")
            self.tts_available = False
        
        # Conversation history for context
        self.conversation_history = []
        
        print("\n🎉 AI Manager initialized successfully!")
        print("Ready to chat!\n")
    
    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribe audio to text using Whisper
        
        Args:
            audio_data: numpy array of audio
            sample_rate: audio sample rate
            
        Returns:
            Transcribed text string or empty string if filtered out
        """
        print("Transcribing audio...")
        
        # List of common Whisper hallucinations to filter out
        HALLUCINATION_PHRASES = [
            'you', 'thank you', 'thanks', 'subtitle', 'subtitles',
            'mbc', 'bbc', 'thank you for watching', 'thanks for watching',
            'please subscribe', 'like and subscribe', 'bye', 'goodbye',
            'see you next time', 'see you', '.', '...'
        ]
        
        try:
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            segments, info = self.whisper.transcribe(
                audio_data,
                language="en",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False  # Prevent hallucinations from context
            )
            
            # Combine all segments
            text = " ".join([segment.text for segment in segments])
            text = text.strip()
            
            # Sanitization: Remove common hallucinations
            text_clean = text.lower().strip().strip('.,!?;:')
            
            if text_clean in HALLUCINATION_PHRASES:
                print(f"Filtered hallucination: '{text}'")
                return ""
            
            # Check minimum length (at least 2 characters)
            if len(text_clean) < 2:
                print(f"Text too short, discarding: '{text}'")
                return ""
            
            print(f"Transcription: {text}")
            return text
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def get_llm_response(self, user_text):
        """
        Get response from Ollama LLM (non-streaming version for compatibility)
        
        Args:
            user_text: User's message
            
        Returns:
            Bot's response text
        """
        print("Getting LLM response...")
        
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_text
            })
            
            # Keep only last 10 messages to save memory
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Get response from Ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=self.conversation_history
            )
            
            bot_response = response['message']['content']
            
            # Add bot response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": bot_response
            })
            
            print(f"Bot response: {bot_response[:100]}...")
            return bot_response
            
        except Exception as e:
            print(f"LLM error: {e}")
            return "I'm sorry, I couldn't process that request."
    
    def get_llm_response_streaming(self, user_text, on_chunk=None, on_sentence=None):
        """
        Get response from Ollama LLM with streaming support
        
        Args:
            user_text: User's message
            on_chunk: Callback for each token chunk (full_text_so_far)
            on_sentence: Callback when a complete sentence is ready (sentence_text)
            
        Returns:
            Bot's full response text
        """
        print("Getting LLM response (streaming)...")
        
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_text
            })
            
            # Keep only last 10 messages to save memory
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Stream response from Ollama
            full_response = ""
            current_sentence = ""
            sentence_delimiters = {'.', '!', '?', '\n'}
            
            stream = ollama.chat(
                model=self.ollama_model,
                messages=self.conversation_history,
                stream=True
            )
            
            for chunk in stream:
                token = chunk['message']['content']
                full_response += token
                current_sentence += token
                
                # Notify about new text chunk for UI update
                if on_chunk:
                    on_chunk(full_response)
                
                # Check if we have a complete sentence
                # Look for sentence-ending punctuation followed by space or end
                for delim in sentence_delimiters:
                    if delim in current_sentence:
                        # Split on delimiter, keeping the delimiter
                        parts = current_sentence.split(delim)
                        
                        # Process complete sentences (all but possibly the last part)
                        for i, part in enumerate(parts[:-1]):
                            complete_sentence = part.strip() + delim
                            if complete_sentence.strip() and len(complete_sentence.strip()) > 1:
                                if on_sentence:
                                    on_sentence(complete_sentence)
                        
                        # Keep the remainder for the next iteration
                        current_sentence = parts[-1]
                        break
            
            # Handle any remaining text as final sentence
            if current_sentence.strip():
                if on_sentence:
                    on_sentence(current_sentence.strip())
            
            # Add bot response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
            print(f"Bot response complete: {full_response[:100]}...")
            return full_response
            
        except Exception as e:
            print(f"LLM error: {e}")
            return "I'm sorry, I couldn't process that request."
    
    def text_to_speech(self, text):
        """
        Convert text to speech using Kokoro ONNX
        
        Args:
            text: Text to synthesize
            
        Returns:
            tuple: (numpy array of audio samples, sample_rate)
        """
        print(f"Generating speech for: '{text[:50]}...'")
        
        if not self.tts_available:
            print("TTS not available, returning silence")
            return np.zeros(24000, dtype=np.float32), 24000
        
        try:
            # Generate audio using Kokoro
            print("Calling Kokoro.create()...")
            samples, sample_rate = self.kokoro.create(text, voice=self.voice_name, speed=1.0)
            
            print(f"Kokoro returned: sample_rate={sample_rate}, samples type={type(samples)}")
            
            # Convert to numpy array if needed
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples, dtype=np.float32)
            
            # Ensure float32
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)
            
            print(f"Generated {len(samples)} audio samples at {sample_rate}Hz")
            print(f"Audio range: [{samples.min():.3f}, {samples.max():.3f}]")
            
            return samples, sample_rate
            
        except Exception as e:
            print(f"TTS error: {e}")
            import traceback
            traceback.print_exc()
            # Return silence as fallback
            return np.zeros(24000, dtype=np.float32), 24000
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared")
