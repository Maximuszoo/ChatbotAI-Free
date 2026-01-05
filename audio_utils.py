"""
Audio utilities for recording and playback with VAD
Handles microphone input and TTS output
"""

import numpy as np
import sounddevice as sd
import queue
import threading
from collections import deque


class AudioRecorder:
    """Records audio with Voice Activity Detection"""
    
    def __init__(self, sample_rate=16000, silence_threshold=0.03, silence_duration=3.0, min_audio_duration=1.0):
        """
        Args:
            sample_rate: Audio sample rate (Hz)
            silence_threshold: RMS threshold for silence detection (increased to avoid noise)
            silence_duration: Seconds of silence before stopping recording
            min_audio_duration: Minimum audio duration in seconds to process
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_audio_duration = min_audio_duration
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.is_paused = False  # For preventing feedback loop
        self.stream = None
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice stream"""
        if status:
            print(f"Audio status: {status}")
        
        if not self.is_paused:
            self.audio_queue.put(indata.copy())
    
    def start_stream(self):
        """Start the audio input stream"""
        # Clear any existing audio in the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=self._audio_callback,
                blocksize=1024
            )
            self.stream.start()
            print("Audio stream started")
    
    def stop_stream(self):
        """Stop the audio input stream"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Audio stream stopped")
    
    def pause_recording(self):
        """Pause recording (to prevent bot hearing itself)"""
        self.is_paused = True
        print("Recording paused")
    
    def resume_recording(self):
        """Resume recording"""
        self.is_paused = False
        # Clear any accumulated audio during pause
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        print("Recording resumed")
    
    def set_silence_duration(self, duration):
        """Update silence duration threshold"""
        self.silence_duration = duration
        print(f"Silence duration set to {duration} seconds")
    
    def record_until_silence(self, max_duration=30):
        """
        Record audio until silence is detected
        
        Args:
            max_duration: Maximum recording duration in seconds
            
        Returns:
            numpy array of recorded audio
        """
        print("Listening for speech...")
        
        audio_buffer = []
        silence_frames = 0
        silence_frames_needed = int(self.silence_duration * self.sample_rate / 1024)
        max_frames = int(max_duration * self.sample_rate / 1024)
        frame_count = 0
        speech_detected = False
        
        # Clear queue before starting
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        while frame_count < max_frames:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(audio_chunk)
                frame_count += 1
                
                # Calculate RMS energy
                rms = np.sqrt(np.mean(audio_chunk ** 2))
                
                if rms > self.silence_threshold:
                    speech_detected = True
                    silence_frames = 0
                else:
                    if speech_detected:
                        silence_frames += 1
                        
                        # Stop if silence detected after speech
                        if silence_frames >= silence_frames_needed:
                            print("Silence detected, stopping recording")
                            break
                            
            except queue.Empty:
                continue
        
        if not audio_buffer:
            return None
        
        # Concatenate all audio chunks
        audio_data = np.concatenate(audio_buffer, axis=0).flatten()
        
        if not speech_detected:
            return None
        
        # Check minimum duration (filter out clicks, breaths, noise)
        audio_duration = len(audio_data) / self.sample_rate
        if audio_duration < self.min_audio_duration:
            print(f"Audio too short ({audio_duration:.2f}s < {self.min_audio_duration}s), discarding")
            return None
        
        print(f"Recorded {audio_duration:.2f}s of audio")
        return audio_data


class AudioPlayer:
    """Plays audio with threading support"""
    
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.is_playing = False
    
    def play(self, audio_data, sample_rate=None):
        """
        Play audio data
        
        Args:
            audio_data: numpy array of audio samples
            sample_rate: override sample rate (optional)
        """
        if sample_rate is not None:
            actual_rate = sample_rate
        else:
            actual_rate = self.sample_rate
            
        self.is_playing = True
        
        try:
            # Ensure audio is float32 and in range [-1, 1]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize if needed
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
                print(f"Normalized audio from max {max_val}")
            
            print(f"Playing audio: {len(audio_data)} samples at {actual_rate}Hz")
            print(f"Duration: {len(audio_data) / actual_rate:.2f} seconds")
            print(f"Audio range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
            
            sd.play(audio_data, actual_rate)
            sd.wait()  # Wait until playback is finished
            
            print("Audio playback finished")
            
        except Exception as e:
            print(f"Playback error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_playing = False
    
    def stop(self):
        """Stop current playback"""
        sd.stop()
        self.is_playing = False
