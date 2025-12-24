import os
import sys

# Try to find the library manually if on Linux
if sys.platform.startswith('linux'):
    # Common locations for libvlc on Linux
    if os.path.exists('/usr/lib/x86_64-linux-gnu/libvlc.so'):
        os.environ['PYTHON_VLC_LIB_PATH'] = '/usr/lib/x86_64-linux-gnu/libvlc.so'
    elif os.path.exists('/usr/lib/libvlc.so'):
        os.environ['PYTHON_VLC_LIB_PATH'] = '/usr/lib/libvlc.so'
    # Also check snap installation
    elif os.path.exists('/snap/vlc/current/usr/lib/x86_64-linux-gnu/libvlc.so'):
        os.environ['PYTHON_VLC_LIB_PATH'] = '/snap/vlc/current/usr/lib/x86_64-linux-gnu/libvlc.so'
    elif os.path.exists('/snap/vlc/current/usr/lib/libvlc.so'):
        os.environ['PYTHON_VLC_LIB_PATH'] = '/snap/vlc/current/usr/lib/libvlc.so'

import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import vlc  # Import vlc AFTER setting the environment variable
import whisper
import json
import subprocess
from pathlib import Path

class SmartVideoPlayer:
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("Smart Video Player & Transcriber")
        self.root.geometry("1600x900")  # Larger window for better video viewing
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.video_path = video_path
        self.segments = [] # Will hold text segments with timestamps
        self.displayed_segments_count = 0  # Track how many segments we've displayed
        self.is_playing = False
        self.player = None
        self.instance = None
        self.transcription_thread = None
        self.transcription_running = False
        self.segments_lock = threading.Lock()  # Lock for thread-safe segment updates

        # --- Layout Configuration ---
        # Video on top, text box below
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)  # Video row
        self.root.rowconfigure(1, weight=0)  # Text box row (fixed height)
        self.root.rowconfigure(2, weight=0)  # Control buttons row

        # --- Video Pane (Top) ---
        self.video_frame = tk.Frame(self.root, bg="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # --- Text Pane (Below Video) ---
        self.text_frame = tk.Frame(self.root, bg="black", height=80)
        self.text_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.text_frame.grid_propagate(False)  # Maintain fixed height
        
        # --- Control Buttons ---
        self.control_frame = tk.Frame(self.root, bg="lightgray")
        self.control_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Play/Pause button
        self.play_pause_btn = tk.Button(self.control_frame, text="⏸ Pause", 
                                       command=self.toggle_play_pause, 
                                       font=("Arial", 12), width=10)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        
        # Stop button
        self.stop_btn = tk.Button(self.control_frame, text="⏹ Stop", 
                                 command=self.stop_video, 
                                 font=("Arial", 12), width=10)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Rewind button (-10 seconds)
        self.rewind_btn = tk.Button(self.control_frame, text="⏪ -10s", 
                                   command=self.rewind_10s, 
                                   font=("Arial", 12), width=10)
        self.rewind_btn.pack(side=tk.LEFT, padx=5)
        
        # Forward button (+10 seconds)
        self.forward_btn = tk.Button(self.control_frame, text="⏩ +10s", 
                                     command=self.forward_10s, 
                                     font=("Arial", 12), width=10)
        self.forward_btn.pack(side=tk.LEFT, padx=5)
        
        # Flashcard button - Capture current sentence
        self.flashcard_btn = tk.Button(self.control_frame, text="📝 Flashcard", 
                                       command=self.capture_flashcard, 
                                       font=("Arial", 12), width=12,
                                       bg="#4CAF50", fg="white")
        self.flashcard_btn.pack(side=tk.LEFT, padx=5)
        
        # View flashcards button
        self.view_flashcards_btn = tk.Button(self.control_frame, text="📚 View Cards", 
                                             command=self.view_flashcards, 
                                             font=("Arial", 12), width=12,
                                             bg="#2196F3", fg="white")
        self.view_flashcards_btn.pack(side=tk.LEFT, padx=5)
        
        # Volume control
        self.volume_label = tk.Label(self.control_frame, text="Volume:", font=("Arial", 10))
        self.volume_label.pack(side=tk.LEFT, padx=(20, 5))
        
        self.volume_scale = tk.Scale(self.control_frame, from_=0, to=100, 
                                     orient=tk.HORIZONTAL, length=150,
                                     command=self.set_volume)
        self.volume_scale.set(70)  # Default volume 70%
        self.volume_scale.pack(side=tk.LEFT, padx=5)
        
        # Seek bar
        self.seek_frame = tk.Frame(self.control_frame)
        self.seek_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.seek_scale = tk.Scale(self.seek_frame, from_=0, to=100, 
                                   orient=tk.HORIZONTAL, length=400,
                                   showvalue=False, resolution=0.1)
        self.seek_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Bind seek events
        self.seek_scale.bind("<Button-1>", self.on_seek_start)
        self.seek_scale.bind("<ButtonRelease-1>", self.on_seek_end)
        self.seek_scale.bind("<B1-Motion>", self.on_seek_drag)
        
        # Time display
        self.time_label = tk.Label(self.control_frame, text="00:00 / 00:00", 
                                   font=("Arial", 10), width=15)
        self.time_label.pack(side=tk.RIGHT, padx=10)
        
        self.seeking = False  # Flag to prevent seek loop
        self.seek_value = 0  # Store seek value during drag
        
        # Flashcard functionality
        self.flashcards = []  # Store flashcards
        self.flashcard_dir = "flashcards"
        os.makedirs(self.flashcard_dir, exist_ok=True)
        self.anki_deck = None  # Will hold the Anki deck
        self.anki_model = None  # Will hold the Anki model
        self.init_anki_deck()  # Initialize Anki deck
        self.load_flashcards()  # Load existing flashcards
        
        # Text Widget - Single line display (larger font for readability)
        self.transcript_box = tk.Text(self.text_frame, wrap=tk.WORD, font=("Arial", 16, "bold"), 
                                     state='disabled', bg="black", fg="white",
                                     height=3, relief=tk.FLAT, bd=0)
        self.transcript_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tags for text styling
        self.transcript_box.tag_config("current", foreground="yellow", font=("Arial", 16, "bold"))
        self.transcript_box.tag_config("timestamp", foreground="gray", font=("Arial", 14))

        # --- Status Bar ---
        self.status_label = tk.Label(self.root, text="Initializing...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=3, column=0, sticky="ew")

        # --- VLC Setup ---
        try:
            # Try with minimal arguments first
            vlc_args = ['--intf', 'dummy', '--quiet']
            self.instance = vlc.Instance(vlc_args)
            
            # Check if instance was created successfully
            if self.instance is None:
                print("Warning: VLC Instance returned None, trying without arguments...")
                self.instance = vlc.Instance()
            
            if self.instance is None:
                raise Exception("Failed to create VLC instance. VLC may not be properly installed.")
            
            self.player = self.instance.media_player_new()
            
            if self.player is None:
                raise Exception("Failed to create VLC media player.")
                
            print("VLC initialized successfully")
            
        except Exception as e:
            error_msg = f"VLC Error: {str(e)}"
            print(error_msg)
            print("\nTroubleshooting:")
            print("1. Make sure VLC is installed: sudo apt-get install vlc")
            print("2. Or: sudo snap install vlc")
            print("3. Check if libvlc.so exists in the paths checked at startup")
            self.update_status(error_msg)
            self.instance = None
            self.player = None
            # Don't return, let the process continue but show error
        
        # Start the heavy lifting in a separate thread so GUI doesn't freeze
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        """
        Setup video player and start real-time transcription.
        """
        # Check for ffmpeg first
        import shutil
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            error_msg = "Error: ffmpeg not found. Please install ffmpeg:\nsudo apt-get install ffmpeg"
            print(error_msg)
            self.update_status("Error: ffmpeg not found")
            self.root.after(0, lambda msg=error_msg: self.show_error(msg))
            return
        
        # Setup VLC media first
        if not self.player or not self.instance:
            error_msg = "Error: VLC not initialized. Cannot play video."
            print(error_msg)
            self.update_status(error_msg)
            self.root.after(0, lambda msg=error_msg: self.show_error(msg))
            return
            
        try:
            media = self.instance.media_new(self.video_path)
            self.player.set_media(media)
            
            # Embed VLC in Tkinter Frame (OS specific)
            win_id = self.video_frame.winfo_id()
            try:
                if os.name == 'nt': # Windows
                    self.player.set_hwnd(win_id)
                else: # Linux/Mac (X11)
                    self.player.set_xwindow(win_id)
            except Exception as e:
                print(f"Warning: Could not embed VLC window: {e}")
                print("VLC will open in a separate window")
            
            # Start video playback immediately
            self.player.play()
            self.is_playing = True
            self.play_pause_btn.config(text="⏸ Pause")
            
            # Set initial volume
            self.player.audio_set_volume(70)
            
            # Start the sync loop and time display
            self.update_status(f"Playing: {os.path.basename(self.video_path)} - Transcribing in real-time...")
            self.root.after(100, self.sync_loop)
            self.root.after(100, self.update_time_display)
            
            # Start real-time transcription in background
            self.transcription_running = True
            self.transcription_thread = threading.Thread(target=self.real_time_transcribe, daemon=True)
            self.transcription_thread.start()
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.update_status(error_msg)
            error_msg_copy = error_msg
            self.root.after(0, lambda msg=error_msg_copy: self.show_error(msg))
    
    def real_time_transcribe(self):
        """Transcribe audio in real-time while video is playing"""
        try:
            print("Loading Whisper model for real-time transcription...")
            # Use device="cpu" and enable threading for multi-core
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("base", device=device)
            print(f"Model loaded on {device}, starting real-time transcription...")
            
            # Use clip_timestamps to process in chunks as video plays
            import numpy as np
            from whisper import load_audio
            
            # Load audio file
            audio = load_audio(self.video_path)
            sample_rate = 16000  # Whisper uses 16kHz
            total_duration = len(audio) / sample_rate
            
            print(f"Video duration: {total_duration:.1f} seconds")
            print("Processing transcription in chunks with multi-core support...")
            
            # Process in smaller chunks (10 seconds) for smoother updates
            chunk_duration = 10.0
            all_segments = []
            processed_time = 0.0
            
            while processed_time < total_duration and self.transcription_running:
                chunk_end = min(processed_time + chunk_duration, total_duration)
                
                # Extract audio chunk
                start_sample = int(processed_time * sample_rate)
                end_sample = int(chunk_end * sample_rate)
                audio_chunk = audio[start_sample:end_sample]
                
                if len(audio_chunk) < sample_rate:  # Less than 1 second, skip
                    break
                
                print(f"Transcribing chunk: {processed_time:.1f}s - {chunk_end:.1f}s")
                
                # Transcribe chunk with optimized settings for speed
                # Use faster settings: greedy decoding (beam_size=1) and lower temperature
                result = model.transcribe(
                    audio_chunk, 
                    verbose=False,
                    condition_on_previous_text=(processed_time > 0),
                    initial_prompt="",
                    temperature=0.0,  # Deterministic, faster
                    beam_size=1,  # Greedy decoding - much faster than beam_size=5
                    best_of=1,  # No sampling, faster
                    fp16=(device == "cuda")  # Use FP16 on GPU for speed
                )
                
                # Adjust timestamps to account for chunk offset
                new_segments = []
                for segment in result['segments']:
                    segment['start'] += processed_time
                    segment['end'] += processed_time
                    new_segments.append(segment)
                    all_segments.append(segment)
                
                # Thread-safe update of segments
                with self.segments_lock:
                    self.segments = sorted(all_segments, key=lambda x: x['start'])
                
                # Segments are updated, sync_loop will display current line
                # No need to update display here - sync_loop handles it
                
                processed_time = chunk_end
                
                # Check current video position - if we're far ahead, slow down processing
                if self.player:
                    try:
                        current_video_time = self.player.get_time() / 1000.0
                        # If we're more than 30 seconds ahead, slow down more
                        if processed_time > current_video_time + 30:
                            time.sleep(0.5)  # Slow down if too far ahead
                        elif processed_time > current_video_time + 15:
                            time.sleep(0.2)  # Moderate delay
                        else:
                            time.sleep(0.05)  # Keep up with video
                    except:
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
            
            print(f"Transcription complete. Found {len(self.segments)} segments.")
            # Final update to ensure all segments are displayed
            with self.segments_lock:
                final_segments = self.segments.copy()
            if len(final_segments) > self.displayed_segments_count:
                self.root.after(0, self.update_transcript_display)
            self.update_status(f"Playing: {os.path.basename(self.video_path)}")
            
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_transcript_display(self):
        """Update the transcript display - this is called when new segments are added"""
        # The actual display is handled by sync_loop which shows current line
        pass
    
    def update_current_text(self):
        """Update the text box to show only the current line being spoken"""
        if not self.player:
            return
            
        try:
            current_time = self.player.get_time() / 1000.0
            
            # Find current segment (only if we're within its time range)
            current_segment = None
            with self.segments_lock:
                for segment in self.segments:
                    # Check if current time is within segment boundaries
                    if segment['start'] <= current_time <= segment['end']:
                        current_segment = segment
                        break
            
            self.transcript_box.config(state='normal')
            self.transcript_box.delete("1.0", tk.END)
            
            if current_segment:
                # Show timestamp and current text
                start_fmt = time.strftime('%M:%S', time.gmtime(current_segment['start']))
                timestamp_text = f"[{start_fmt}] "
                text = current_segment['text'].strip()
                
                self.transcript_box.insert("1.0", timestamp_text, "timestamp")
                self.transcript_box.insert(tk.END, text, "current")
            else:
                # No current segment - check if we're in a gap between segments
                # Find the most recent segment that has ended
                most_recent_segment = None
                with self.segments_lock:
                    for segment in reversed(self.segments):
                        if segment['end'] < current_time:
                            most_recent_segment = segment
                            break
                
                if most_recent_segment and (current_time - most_recent_segment['end']) < 5.0:
                    # Show the most recent segment if we're within 5 seconds of it ending
                    start_fmt = time.strftime('%M:%S', time.gmtime(most_recent_segment['start']))
                    self.transcript_box.insert("1.0", f"[{start_fmt}] {most_recent_segment['text'].strip()}", "current")
                elif not self.segments:
                    # No segments yet
                    self.transcript_box.insert("1.0", "Transcribing...", "current")
                else:
                    # In a gap - show empty or "..." to indicate silence
                    self.transcript_box.insert("1.0", "...", "timestamp")
            
            self.transcript_box.config(state='disabled')
        except Exception as e:
            # Silently handle errors to avoid spam
            pass
    
    def show_error(self, error_msg):
        """Display error message in text box"""
        self.transcript_box.config(state='normal')
        self.transcript_box.delete("1.0", tk.END)
        self.transcript_box.insert("1.0", f"Error occurred:\n\n{error_msg}\n\nPlease check the console for details.")
        self.transcript_box.config(state='disabled')
    
    def capture_flashcard(self):
        """Capture current sentence and create a flashcard"""
        if not self.player:
            messagebox.showwarning("No Video", "No video is currently playing.")
            return
        
        try:
            current_time = self.player.get_time() / 1000.0
            
            # Find current segment
            current_segment = None
            with self.segments_lock:
                for segment in self.segments:
                    if segment['start'] <= current_time <= segment['end']:
                        current_segment = segment
                        break
            
            if not current_segment:
                messagebox.showinfo("No Speech", "No speech detected at current position.")
                return
            
            sentence = current_segment['text'].strip()
            start_time = current_segment['start']
            end_time = current_segment['end']
            
            # Show processing message
            self.update_status("Creating flashcard...")
            
            # Generate flashcard in background thread
            threading.Thread(target=self.generate_flashcard, 
                           args=(sentence, start_time, end_time, current_time),
                           daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture flashcard: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_flashcard(self, sentence, start_time, end_time, video_time):
        """Generate flashcard content using offline AI (runs in background thread)"""
        try:
            # Update status on main thread
            self.root.after(0, lambda: self.update_status("Extracting audio..."))
            
            # Extract audio for this segment (0.2 seconds before and after)
            audio_start = max(0, start_time - 0.2)
            audio_end = end_time + 0.2
            audio_path = self.extract_audio_segment(audio_start, audio_end)
            
            # Update status on main thread
            self.root.after(0, lambda: self.update_status("Extracting video scene as GIF..."))
            
            # Extract video scene as optimized GIF (small file size)
            gif_start = max(0, start_time - 0.2)
            gif_end = end_time + 0.2
            gif_path = self.extract_video_gif_optimized(gif_start, gif_end)
            
            # Update status on main thread
            self.root.after(0, lambda: self.update_status("Generating flashcard content..."))
            
            # Generate flashcard content
            flashcard_data = self.create_flashcard_content(sentence, audio_path, gif_path, video_time)
            
            # Prepare flashcard data
            flashcard_id = len(self.flashcards)
            flashcard_data['id'] = flashcard_id
            flashcard_data['timestamp'] = video_time
            flashcard_data['sentence'] = sentence
            flashcard_data['audio_path'] = audio_path
            flashcard_data['gif_path'] = gif_path
            
            # Now switch to main thread for dialog and saving
            # This prevents freezing because dialog must run on main thread
            self.root.after(0, lambda: self._show_flashcard_dialog_and_save(flashcard_data, flashcard_id))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.update_status("Error creating flashcard"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate flashcard: {error_msg}"))
            import traceback
            traceback.print_exc()
    
    def _show_flashcard_dialog_and_save(self, flashcard_data, flashcard_id):
        """Show edit dialog and save flashcard (runs on main thread)"""
        try:
            # Show edit dialog before saving (must be on main thread)
            edited_data = self.edit_flashcard_dialog(flashcard_data)
            if edited_data is None:
                # User cancelled
                self.update_status("Flashcard creation cancelled")
                return
            
            # Use edited data
            flashcard_data = edited_data
            
            # Update status
            self.update_status("Saving flashcard to Anki deck...")
            
            # Save in background thread to avoid blocking UI
            threading.Thread(
                target=self._save_flashcard_background,
                args=(flashcard_data, flashcard_id),
                daemon=True
            ).start()
            
        except Exception as e:
            error_msg = str(e)
            self.update_status("Error saving flashcard")
            messagebox.showerror("Error", f"Failed to save flashcard: {error_msg}")
            import traceback
            traceback.print_exc()
    
    def _save_flashcard_background(self, flashcard_data, flashcard_id):
        """Save flashcard in background thread"""
        try:
            self.flashcards.append(flashcard_data)
            self.save_flashcard(flashcard_data)
            
            # Show success message on main thread
            deck_path = Path(self.flashcard_dir) / "Video_Learning_Flashcards.apkg"
            self.root.after(0, lambda: messagebox.showinfo("Flashcard Created", 
                f"Flashcard #{flashcard_id + 1} created successfully!\n\n"
                f"Total cards in deck: {len(self.flashcards)}\n\n"
                f"Anki deck package:\n{deck_path.name}\n\n"
                f"Location: {deck_path.parent}\n\n"
                f"Import this .apkg file into Anki to use all flashcards at once!"))
            self.root.after(0, lambda: self.update_status(f"Flashcard #{flashcard_id + 1} created - Deck has {len(self.flashcards)} cards"))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.update_status("Error saving flashcard"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to save flashcard: {error_msg}"))
            import traceback
            traceback.print_exc()
    
    def extract_audio_segment(self, start_time, end_time):
        """Extract audio segment from video"""
        try:
            from pathlib import Path
            temp_dir = Path(self.flashcard_dir) / "audio"
            temp_dir.mkdir(exist_ok=True)
            
            # Create unique filename
            audio_filename = f"flashcard_{int(start_time)}_{int(end_time)}.wav"
            audio_path = temp_dir / audio_filename
            
            # Use ffmpeg to extract audio segment
            cmd = [
                'ffmpeg', '-i', self.video_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',  # Overwrite output file
                str(audio_path)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            return str(audio_path)
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def extract_video_gif_optimized(self, start_time, end_time):
        """Extract video segment and convert to optimized GIF (small file size, Anki-compatible)"""
        try:
            import cv2
            import os
            from PIL import Image
            
            from pathlib import Path
            temp_dir = Path(self.flashcard_dir) / "gifs"
            temp_dir.mkdir(exist_ok=True)
            
            # Create unique filename
            gif_filename = f"flashcard_{int(start_time)}_{int(end_time)}.gif"
            gif_path = temp_dir / gif_filename
            
            # Open video file
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print("Error: Could not open video file")
                return None
            
            # Get video FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 25  # Default FPS if unknown
            
            # Calculate frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            total_frames = end_frame - start_frame
            
            # Very aggressive optimization: limit to max 6 frames for smallest file size
            max_frames = 6
            if total_frames > max_frames:
                frame_skip = max(1, total_frames // max_frames)
            else:
                frame_skip = 1
            
            # Seek to start position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            frame_count = 0
            
            print(f"Extracting optimized GIF: {start_time:.1f}s - {end_time:.1f}s")
            
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames aggressively to keep file size small
                if frame_count % frame_skip == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Very small dimensions to reduce file size (max width 320px)
                    height, width = frame_rgb.shape[:2]
                    if width > 320:
                        scale = 320 / width
                        new_width = 320
                        new_height = int(height * scale)
                        frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    frames.append(frame_rgb)
                
                frame_count += 1
                
                # Check if we've reached the end
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if current_frame >= end_frame:
                    break
            
            cap.release()
            
            if not frames:
                print("Error: No frames extracted")
                return None
            
            # Use PIL to create proper GIF89a format that Anki can display
            # Very low FPS (2-3 fps) for small file size
            duration = (end_time - start_time) / len(frames) if len(frames) > 0 else 0.1
            gif_fps = min(3, 1.0 / duration) if duration > 0 else 3  # Max 3 fps
            frame_duration_ms = int(1000 / gif_fps)  # Duration in milliseconds
            
            # Convert frames to PIL Images with optimized palette
            import numpy as np
            pil_frames = []
            
            for frame in frames:
                # Convert to PIL Image
                img = Image.fromarray(frame.astype('uint8'))
                # Quantize to 64 colors for smaller file size
                # Use ADAPTIVE palette for best quality at low color count
                img_p = img.convert('P', palette=Image.ADAPTIVE, colors=64)
                pil_frames.append(img_p)
            
            if not pil_frames:
                print("Error: No frames to save")
                return None
            
            # Save as GIF89a format using PIL - this is what Anki needs
            # Anki requires proper GIF89a format with correct frame settings
            # Note: Some older Anki versions may not support animated GIFs
            # Make sure you're using Anki 2.1.50+ or latest mobile versions
            try:
                pil_frames[0].save(
                    str(gif_path),
                    format='GIF',
                    save_all=True,
                    append_images=pil_frames[1:] if len(pil_frames) > 1 else [],
                    duration=frame_duration_ms,
                    loop=0,  # Loop forever (0 = infinite loop)
                    optimize=False,  # Disable optimize for better Anki compatibility
                    disposal=2  # Restore to background color (works better in Anki)
                )
            except Exception as e:
                print(f"Error saving GIF with PIL: {e}")
                # Fallback: try without disposal parameter
                pil_frames[0].save(
                    str(gif_path),
                    format='GIF',
                    save_all=True,
                    append_images=pil_frames[1:] if len(pil_frames) > 1 else [],
                    duration=frame_duration_ms,
                    loop=0
                )
            
            file_size_kb = os.path.getsize(gif_path) / 1024
            print(f"Optimized GIF saved: {gif_path} ({len(frames)} frames, {gif_fps:.1f} fps, {file_size_kb:.1f} KB)")
            return str(gif_path)
            
        except Exception as e:
            print(f"Error extracting optimized GIF: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_flashcard_content(self, sentence, audio_path, gif_path, video_time):
        """Create flashcard content using offline AI analysis"""
        try:
            # Get phonetic transcription for the entire sentence
            phonetic_transcription = self.get_sentence_phonetic(sentence)
            
            return {
                'front': {
                    'sentence': sentence,
                    'audio_path': audio_path,
                    'gif_path': gif_path,
                    'timestamp': f"{int(video_time // 60):02d}:{int(video_time % 60):02d}"
                },
                'back': {
                    'phonetic': phonetic_transcription
                }
            }
        except Exception as e:
            print(f"Error creating flashcard content: {e}")
            import traceback
            traceback.print_exc()
            # Return basic structure if analysis fails
            return {
                'front': {
                    'sentence': sentence,
                    'audio_path': audio_path,
                    'gif_path': gif_path,
                    'timestamp': f"{int(video_time // 60):02d}:{int(video_time % 60):02d}"
                },
                'back': {
                    'phonetic': sentence  # Fallback to sentence if phonetic fails
                }
            }
    
    def get_sentence_phonetic(self, sentence):
        """Get phonetic transcription for the entire sentence"""
        try:
            import re
            words = re.findall(r'\b\w+\b', sentence)
            phonetic_words = []
            
            for word in words:
                phonetic = self.get_pronunciation(word)
                phonetic_words.append(phonetic)
            
            return ' '.join(phonetic_words)
        except Exception as e:
            print(f"Error getting sentence phonetic: {e}")
            return sentence
    
    def translate_to_persian(self, sentence):
        """Translate sentence to Persian (offline)"""
        try:
            # Try using argostranslate for offline translation
            try:
                import argostranslate.package
                import argostranslate.translate
                
                # Check if English to Persian package is installed
                installed_languages = argostranslate.translate.get_installed_languages()
                from_code = None
                to_code = None
                
                for lang in installed_languages:
                    if lang.code == 'en':
                        from_code = lang
                    elif lang.code == 'fa':
                        to_code = lang
                
                if from_code and to_code:
                    translation = argostranslate.translate.translate(sentence, from_code, to_code)
                    return translation
                else:
                    print("Warning: English-Persian translation package not installed.")
                    print("Install with: argostranslate.package.update_package_index()")
                    print("Then: argostranslate.package.install_from_path(argostranslate.package.get_available_packages()[0].download())")
                    return self.simple_persian_translation(sentence)
            except ImportError:
                # Fallback to simple method
                return self.simple_persian_translation(sentence)
            except Exception as e:
                print(f"Translation error: {e}")
                return self.simple_persian_translation(sentence)
        except Exception as e:
            print(f"Error in translation: {e}")
            return self.simple_persian_translation(sentence)
    
    def simple_persian_translation(self, sentence):
        """Simple placeholder for Persian translation"""
        # This is a placeholder - in production, use argostranslate or another offline translator
        return f"[ترجمه فارسی: {sentence}]"  # Placeholder text
    
    def get_pronunciation(self, word):
        """Get pronunciation guide for a word using IPA (International Phonetic Alphabet)"""
        try:
            # Try to use eng-to-ipa if available
            try:
                import eng_to_ipa as ipa
                pronunciation = ipa.convert(word, retrieve_all=False)
                if pronunciation and pronunciation != word:
                    return pronunciation
                # If conversion failed, try with stress marks
                pronunciation = ipa.convert(word, retrieve_all=True)
                if isinstance(pronunciation, list) and pronunciation:
                    return pronunciation[0]
                return pronunciation if pronunciation and pronunciation != word else self.simple_phonetic(word)
            except ImportError:
                # Fallback: simple phonetic approximation
                return self.simple_phonetic(word)
            except Exception as e:
                print(f"Error getting pronunciation for '{word}': {e}")
                return self.simple_phonetic(word)
        except:
            return word
    
    def simple_phonetic(self, word):
        """Simple phonetic approximation (fallback when eng-to-ipa is not available)"""
        word = word.lower()
        phonetic = word
        if len(word) > 1:
            if word[0] in 'aeiou':
                phonetic = f"'{word}"
            else:
                phonetic = f"'{word[0]}-{word[1:]}"
        return phonetic
    
    def analyze_grammar(self, sentence):
        """Analyze grammar points in the sentence using pattern matching"""
        grammar_points = []
        try:
            import re
            sentence_lower = sentence.lower()
            
            # Past tense
            past_tense_patterns = [
                r'\b(was|were|did|had|went|came|said|told|thought|knew|saw|heard|felt)\b',
                r'\b\w+ed\b',
                r'\b(ran|swam|ate|drank|wrote|spoke|broke|chose|froze|threw)\b'
            ]
            if any(re.search(pattern, sentence_lower) for pattern in past_tense_patterns):
                grammar_points.append("Past tense: Used to describe completed actions in the past")
            
            # Present continuous
            if re.search(r'\b(is|are|am|was|were)\s+\w+ing\b', sentence_lower):
                grammar_points.append("Present/Past continuous: Actions happening at a specific time")
            
            # Future tense
            if re.search(r'\b(will|shall|going to|gonna)\b', sentence_lower):
                grammar_points.append("Future tense: Expressing future plans or predictions")
            
            # Perfect tenses
            if re.search(r'\b(has|have|had)\s+\w+ed\b', sentence_lower):
                grammar_points.append("Perfect tense: Actions completed before a point in time")
            
            # Conditional
            if re.search(r'\b(if|would|could|should|might|may)\b', sentence_lower):
                grammar_points.append("Conditional/Modal verbs: Expressing possibility, ability, or hypothetical situations")
            
            # Passive voice
            if re.search(r'\b(was|were|is|are|been)\s+\w+ed\b', sentence_lower):
                grammar_points.append("Passive voice: When the subject receives the action")
            
            # Questions
            if sentence.strip().endswith('?'):
                if re.search(r'\b(do|does|did|is|are|was|were|have|has|can|could|will|would)\b', sentence_lower):
                    grammar_points.append("Question formation: Using auxiliary verbs to form questions")
                else:
                    grammar_points.append("Question formation: Wh-questions or tag questions")
            
            # Relative clauses
            if re.search(r'\b(who|which|that|where|when|whom|whose)\b', sentence_lower):
                grammar_points.append("Relative clauses: Adding extra information about nouns")
            
        except Exception as e:
            print(f"Error in grammar analysis: {e}")
        
        return grammar_points if grammar_points else ["Standard English sentence structure"]
    
    def generate_examples(self, sentence, important_words):
        """Generate example sentences using important words"""
        examples = []
        try:
            for word in important_words[:3]:
                if word.endswith('ing'):
                    examples.append(f"• '{word}' (verb + -ing): \"I enjoy {word} in my free time.\"")
                elif word.endswith('ed'):
                    examples.append(f"• '{word}' (past tense): \"Yesterday, I {word} the project.\"")
                elif word.endswith('ly'):
                    examples.append(f"• '{word}' (adverb): \"She spoke {word} during the meeting.\"")
                elif word.endswith('tion') or word.endswith('sion'):
                    examples.append(f"• '{word}' (noun): \"The {word} was very important.\"")
                else:
                    examples.append(f"• '{word}': \"Can you explain what '{word}' means in this context?\"")
            
            if important_words:
                examples.append(f"\n💡 Practice tip: Try using these words ({', '.join(important_words[:3])}) in your own sentences to remember them better.")
        except Exception as e:
            print(f"Error generating examples: {e}")
        
        return examples if examples else ["💡 Practice using these words in your own sentences to improve your vocabulary."]
    
    def save_flashcard(self, flashcard_data):
        """Save flashcard to JSON file and add to Anki deck"""
        try:
            # Save as JSON for internal use
            flashcard_file = Path(self.flashcard_dir) / f"flashcard_{flashcard_data['id']}.json"
            with open(flashcard_file, 'w', encoding='utf-8') as f:
                json.dump(flashcard_data, f, indent=2, ensure_ascii=False)
            
            # Add to Anki deck
            self.add_to_anki_deck(flashcard_data)
            
        except Exception as e:
            print(f"Error saving flashcard: {e}")
            import traceback
            traceback.print_exc()
    
    def init_anki_deck(self):
        """Initialize Anki deck with a Professional Template"""
        try:
            import genanki
            
            # CSS: Modern, Clean, Dark-Mode Compatible
            # Defines the look of the card
            professional_css = """
            .card {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 16px;
                text-align: center;
                color: #333;
                background-color: #f0f2f5; /* Light grey background for the window */
                height: 100%;
                margin: 0;
                padding: 10px;
            }
            
            /* The White Container Card */
            .flashcard-container {
                background-color: #ffffff;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                max-width: 600px;
                margin: 20px auto;
                overflow: hidden;
                border: 1px solid #e1e4e8;
            }

            /* Dark Mode Support for Anki */
            .nightMode .card { background-color: #1a1a1a; color: #f0f0f0; }
            .nightMode .flashcard-container { background-color: #2b2b2b; border-color: #404040; }
            .nightMode .label { color: #aaa; }
            .nightMode .phonetic-box { background-color: #333; color: #4dabf7; }

            /* Visual Elements */
            .media-area {
                padding: 0;
                background-color: #000;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 200px;
            }
            
            .media-area img {
                width: 100%;
                height: auto;
                display: block;
                max-height: 350px;
                object-fit: contain;
            }

            .content-area {
                padding: 25px;
                text-align: left;
            }

            .label {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #6c757d;
                margin-bottom: 5px;
                font-weight: 600;
                display: block;
            }

            .sentence-text {
                font-size: 1.6rem;
                line-height: 1.4;
                font-weight: 600;
                color: #2d3436;
                margin-bottom: 20px;
            }
            
            .nightMode .sentence-text { color: #e0e0e0; }

            .phonetic-box {
                display: inline-block;
                background-color: #e3f2fd;
                color: #1976d2;
                padding: 6px 12px;
                border-radius: 6px;
                font-family: 'Courier New', monospace;
                font-size: 0.95rem;
                margin-top: 5px;
            }

            .audio-container {
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #eee;
                text-align: center;
            }
            
            .nightMode .audio-container { border-top: 1px solid #444; }

            /* Hidden audio player fix for AnkiDesktop */
            .replay-button svg { width: 24px; height: 24px; fill: #1976d2; }
            """

            # HTML: Front of Card (What you see first)
            front_template = """
            <div class="flashcard-container">
                <div class="media-area">
                    {{Image}}
                </div>
                <div class="content-area" style="text-align: center;">
                    <span class="label">Listen to the clip</span>
                    <div class="audio-container">
                        {{Audio}}
                    </div>
                </div>
            </div>
            """

            # HTML: Back of Card (The reveal)
            back_template = """
            <div class="flashcard-container">
                <div class="media-area">
                    {{Image}}
                </div>
                
                <div class="content-area">
                    <span class="label">Transcript</span>
                    <div class="sentence-text">{{Sentence}}</div>
                    
                    <span class="label">Pronunciation (IPA)</span>
                    <div class="phonetic-box">{{Phonetic}}</div>
                    
                    <div class="audio-container">
                        {{Audio}}
                    </div>
                </div>
            </div>
            """

            # Create the Model (fixed IDs for updatable deck)
            # IMPORTANT: Keep these IDs constant so the deck can be updated
            my_model = genanki.Model(
                1607392319,  # Fixed model ID - DO NOT CHANGE
                'Smart Video Card Pro',
                fields=[
                    {'name': 'Image'},
                    {'name': 'Audio'},
                    {'name': 'Sentence'},
                    {'name': 'Phonetic'},
                ],
                templates=[
                    {
                        'name': 'Card 1',
                        'qfmt': front_template,
                        'afmt': back_template,
                    },
                ],
                css=professional_css
            )
            
            # Create deck (fixed ID for updatable deck)
            # IMPORTANT: Keep this ID constant so the deck can be updated
            self.anki_deck = genanki.Deck(
                2059400110,  # Fixed deck ID - DO NOT CHANGE
                'Video Learning Flashcards'
            )
            
            self.anki_model = my_model
            print("Anki deck initialized with Professional Template")
            
        except ImportError:
            print("Warning: genanki not installed. Install with: pip install genanki")
            self.anki_deck = None
        except Exception as e:
            print(f"Error initializing Anki deck: {e}")
            self.anki_deck = None
    
    def add_to_anki_deck(self, flashcard_data):
        """Add flashcard to Anki deck"""
        try:
            if not self.anki_deck:
                self.init_anki_deck()
            
            if not self.anki_deck:
                return  # Can't create deck
            
            import genanki
            
            # Format the back content
            back_data = flashcard_data.get('back', {})
            
            # Get phonetic transcription
            phonetic_text = back_data.get('phonetic', '')
            if not phonetic_text:
                # Generate if missing
                phonetic_text = self.get_sentence_phonetic(flashcard_data.get('sentence', ''))
            
            # Debug output
            print(f"Flashcard back content:")
            print(f"  Sentence: {flashcard_data.get('sentence', '')}")
            print(f"  Phonetic: {phonetic_text}")
            
            # Format GIF (if available)
            gif_html = ""
            if flashcard_data.get('gif_path') and os.path.exists(flashcard_data['gif_path']):
                # For Anki, we need to reference the GIF file
                gif_filename = os.path.basename(flashcard_data['gif_path'])
                # Anki requires simple img tag - no complex styling that might break animation
                gif_html = f'<img src="{gif_filename}">'
                
                # Copy GIF to Anki media folder
                anki_media_dir = Path(self.flashcard_dir) / "anki_media"
                anki_media_dir.mkdir(exist_ok=True)
                import shutil
                dest_gif = anki_media_dir / gif_filename
                shutil.copy2(flashcard_data['gif_path'], dest_gif)
                
                # Verify GIF is valid
                try:
                    from PIL import Image
                    test_img = Image.open(dest_gif)
                    if hasattr(test_img, 'is_animated') and test_img.is_animated:
                        print(f"GIF verified: {gif_filename} is animated ({test_img.n_frames} frames)")
                    else:
                        print(f"Warning: GIF {gif_filename} may not be animated")
                    test_img.close()
                except Exception as e:
                    print(f"Warning: Could not verify GIF format: {e}")
            
            # Format audio (if available)
            audio_html = ""
            if flashcard_data.get('audio_path') and os.path.exists(flashcard_data['audio_path']):
                # For Anki, we need to reference the audio file
                # We'll copy it to the media folder and reference it
                audio_filename = os.path.basename(flashcard_data['audio_path'])
                audio_html = f'[sound:{audio_filename}]'
                
                # Copy audio to Anki media folder
                anki_media_dir = Path(self.flashcard_dir) / "anki_media"
                anki_media_dir.mkdir(exist_ok=True)
                import shutil
                dest_audio = anki_media_dir / audio_filename
                shutil.copy2(flashcard_data['audio_path'], dest_audio)
            
            # Create Anki note with unique GUID for each card (allows updates)
            # Use stable GUID based on flashcard ID and sentence for updatability
            # This allows Anki to recognize and update existing cards when re-importing
            note = genanki.Note(
                model=self.anki_model,
                fields=[
                    gif_html,  # GIF
                    audio_html,  # Audio
                    flashcard_data['sentence'],  # Sentence
                    phonetic_text,  # Phonetic transcription
                ],
                guid=genanki.guid_for(flashcard_data.get('id', 0), flashcard_data.get('sentence', ''))
            )
            
            self.anki_deck.add_note(note)
            
            # Save Anki deck after each card
            self.export_anki_deck()
            
        except Exception as e:
            print(f"Error adding to Anki deck: {e}")
            import traceback
            traceback.print_exc()
    
    def export_anki_deck(self):
        """Export Anki deck to .apkg file"""
        try:
            if not self.anki_deck:
                return
            
            import genanki
            
            # Create package with media files
            anki_media_dir = Path(self.flashcard_dir) / "anki_media"
            media_files = []
            if anki_media_dir.exists():
                # Include both audio and GIF files
                media_files = [str(f) for f in anki_media_dir.glob("*.wav")]
                media_files.extend([str(f) for f in anki_media_dir.glob("*.gif")])
            
            # Save deck - all flashcards in one .apkg file
            deck_path = Path(self.flashcard_dir) / "Video_Learning_Flashcards.apkg"
            package = genanki.Package(self.anki_deck, media_files)
            package.write_to_file(str(deck_path))
            
            print(f"Anki deck exported to: {deck_path}")
            print(f"  Total cards: {len(self.anki_deck.notes)}")
            print(f"  Media files: {len(media_files)}")
            print(f"  All flashcards are packaged in one file: {deck_path.name}")
            
        except Exception as e:
            print(f"Error exporting Anki deck: {e}")
            import traceback
            traceback.print_exc()
    
    def load_flashcards(self):
        """Load flashcards from disk and rebuild Anki deck"""
        try:
            flashcard_files = sorted(Path(self.flashcard_dir).glob("flashcard_*.json"))
            for file in flashcard_files:
                with open(file, 'r', encoding='utf-8') as f:
                    flashcard = json.load(f)
                    self.flashcards.append(flashcard)
            
            # Rebuild Anki deck from loaded flashcards
            if self.flashcards and self.anki_deck:
                # Clear existing deck and rebuild (using same IDs for updatability)
                self.init_anki_deck()
                for flashcard in self.flashcards:
                    self.add_to_anki_deck(flashcard)
                print(f"Rebuilt Anki deck with {len(self.flashcards)} cards")
                print("Deck is updatable - you can add/remove cards and re-import to Anki")
        except Exception as e:
            print(f"Error loading flashcards: {e}")
            import traceback
            traceback.print_exc()
    
    def view_flashcards(self):
        """Open flashcard viewer window"""
        if not self.flashcards:
            messagebox.showinfo("No Flashcards", "No flashcards have been created yet.\n\nClick the '📝 Flashcard' button while watching to create one.")
            return
        
        # Create flashcard viewer window
        viewer = tk.Toplevel(self.root)
        viewer.title(f"Flashcards ({len(self.flashcards)} cards)")
        viewer.geometry("900x700")
        
        # Create notebook for flashcard navigation
        notebook = ttk.Notebook(viewer)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for i, card in enumerate(self.flashcards):
            frame = tk.Frame(notebook)
            notebook.add(frame, text=f"Card {i+1}")
            
            # Front side
            front_frame = tk.LabelFrame(frame, text="Front", font=("Arial", 14, "bold"))
            front_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            sentence_label = tk.Label(front_frame, text=card['sentence'], 
                                     font=("Arial", 16), wraplength=800, justify=tk.LEFT)
            sentence_label.pack(pady=20, padx=20)
            
            timestamp_label = tk.Label(front_frame, text=f"Time: {card.get('timestamp', 'N/A')}", 
                                      font=("Arial", 12), fg="gray")
            timestamp_label.pack(pady=5)
            
            # Audio playback button
            if card.get('audio_path') and os.path.exists(card['audio_path']):
                audio_btn = tk.Button(front_frame, text="🔊 Play Audio", 
                                     command=lambda path=card['audio_path']: self.play_audio(path),
                                     font=("Arial", 12))
                audio_btn.pack(pady=10)
            
            # Back side
            back_frame = tk.LabelFrame(frame, text="Back", font=("Arial", 14, "bold"))
            back_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            back_text = scrolledtext.ScrolledText(back_frame, wrap=tk.WORD, 
                                                  font=("Arial", 12), height=15)
            back_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            back_data = card.get('back', {})
            
            # Add pronunciations
            if back_data.get('pronunciations'):
                back_text.insert(tk.END, "📢 PRONUNCIATION:\n", "bold")
                back_text.tag_config("bold", font=("Arial", 12, "bold"))
                for word, pron in back_data['pronunciations'].items():
                    back_text.insert(tk.END, f"  • {word}: /{pron}/\n\n")
            
            # Add grammar points
            if back_data.get('grammar_points'):
                back_text.insert(tk.END, "📚 GRAMMAR POINTS:\n", "bold")
                for point in back_data['grammar_points']:
                    back_text.insert(tk.END, f"  • {point}\n\n")
            
            # Add examples
            if back_data.get('examples'):
                back_text.insert(tk.END, "💡 EXAMPLES:\n", "bold")
                for example in back_data['examples']:
                    back_text.insert(tk.END, f"  • {example}\n\n")
            
            back_text.config(state='disabled')
    
    def play_audio(self, audio_path):
        """Play audio file"""
        try:
            if sys.platform.startswith('linux'):
                subprocess.Popen(['ffplay', '-nodisp', '-autoexit', audio_path], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sys.platform == 'darwin':
                subprocess.Popen(['afplay', audio_path])
            elif sys.platform == 'win32':
                subprocess.Popen(['powershell', '-c', f'(New-Object Media.SoundPlayer "{audio_path}").PlaySync()'])
        except Exception as e:
            print(f"Error playing audio: {e}")
            messagebox.showerror("Audio Error", f"Could not play audio: {e}")

    def populate_text(self):
        """Fills the text box with the transcribed text."""
        self.update_transcript_display()
    
    def edit_flashcard_dialog(self, flashcard_data):
        """Show edit dialog for flashcard before saving"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Flashcard")
        dialog.geometry("700x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        result = {'cancelled': False, 'data': None}
        
        # Sentence field
        tk.Label(dialog, text="Sentence:", font=("Arial", 12, "bold")).pack(pady=(10, 5), padx=20, anchor='w')
        sentence_text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, height=4, font=("Arial", 12))
        sentence_text.pack(fill=tk.BOTH, expand=False, padx=20, pady=5)
        sentence_text.insert("1.0", flashcard_data.get('sentence', ''))
        
        # Phonetic field
        tk.Label(dialog, text="Phonetic Transcription:", font=("Arial", 12, "bold")).pack(pady=(10, 5), padx=20, anchor='w')
        phonetic_text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, height=3, font=("Courier New", 11))
        phonetic_text.pack(fill=tk.BOTH, expand=False, padx=20, pady=5)
        back_data = flashcard_data.get('back', {})
        phonetic_text.insert("1.0", back_data.get('phonetic', ''))
        
        # Persian translation field
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=20, padx=20, fill=tk.X)
        
        def save_and_close():
            # Get edited values
            edited_sentence = sentence_text.get("1.0", tk.END).strip()
            edited_phonetic = phonetic_text.get("1.0", tk.END).strip()
            
            if not edited_sentence:
                messagebox.showwarning("Empty Sentence", "Sentence cannot be empty!")
                return
            
            # Update flashcard data
            flashcard_data['sentence'] = edited_sentence
            flashcard_data['back'] = {
                'phonetic': edited_phonetic
            }
            
            result['data'] = flashcard_data
            dialog.destroy()
        
        def cancel():
            result['cancelled'] = True
            dialog.destroy()
        
        save_btn = tk.Button(button_frame, text="Save Flashcard", command=save_and_close,
                            font=("Arial", 12), bg="#4CAF50", fg="white", width=15)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=cancel,
                              font=("Arial", 12), bg="#f44336", fg="white", width=15)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Wait for dialog to close
        dialog.wait_window()
        
        if result['cancelled']:
            return None
        
        return result['data']

    def sync_loop(self):
        """Checks video time and highlights relevant text."""
        if not self.player:
            return
            
        try:
            # Get current time in seconds
            current_time = self.player.get_time() / 1000.0
            
            # Check if video has ended
            state = self.player.get_state()
            if state == vlc.State.Ended:
                self.is_playing = False
                self.play_pause_btn.config(text="▶ Play")
                self.update_status("Video ended")
                return
            
            # Update playing state based on VLC state
            if state == vlc.State.Playing:
                self.is_playing = True
                self.play_pause_btn.config(text="⏸ Pause")
            elif state == vlc.State.Paused:
                self.is_playing = False
                self.play_pause_btn.config(text="▶ Play")
            
            # Update current text display (single line)
            self.update_current_text()
            
            # Run again in 500ms
            self.root.after(500, self.sync_loop)
        except Exception as e:
            # Video might have ended or player stopped
            error_str = str(e)
            if "Invalid" not in error_str and "NoneType" not in error_str:
                print(f"Sync error: {e}")
            # Stop the loop if there's a persistent error
            if "NoneType" in error_str:
                self.is_playing = False
                return
            # Try to continue
            self.root.after(500, self.sync_loop)
    
    def update_time_display(self):
        """Update the time display label and seek bar"""
        if not self.player:
            return
            
        try:
            current_time_ms = self.player.get_time()
            total_time_ms = self.player.get_length()
            
            if current_time_ms >= 0 and total_time_ms > 0:
                current_sec = current_time_ms // 1000
                total_sec = total_time_ms // 1000
                
                current_str = f"{current_sec // 60:02d}:{current_sec % 60:02d}"
                total_str = f"{total_sec // 60:02d}:{total_sec % 60:02d}"
                
                self.time_label.config(text=f"{current_str} / {total_str}")
                
                # Update seek bar (only if not currently seeking)
                if not self.seeking:
                    position = max(0, min(100, (current_time_ms / total_time_ms) * 100))
                    self.seek_scale.set(position)
            
            # Update every 200ms for smoother updates
            self.root.after(200, self.update_time_display)
        except:
            # If error, try again
            self.root.after(200, self.update_time_display)
    
    def on_seek_start(self, event):
        """Handle seek bar drag start"""
        self.seeking = True
    
    def on_seek_drag(self, event):
        """Handle seek bar dragging - update preview but don't seek yet"""
        # Just track the value, actual seek happens on release
        pass
    
    def on_seek_end(self, event):
        """Handle seek bar drag end - perform the seek"""
        if not self.player:
            self.seeking = False
            return
            
        try:
            total_time_ms = self.player.get_length()
            if total_time_ms > 0:
                value = self.seek_scale.get()
                seek_position = int((float(value) / 100.0) * total_time_ms)
                # Ensure seek position is valid
                seek_position = max(0, min(seek_position, total_time_ms))
                self.player.set_time(seek_position)
                print(f"Seeking to: {seek_position}ms ({seek_position/1000:.1f}s, {value:.1f}%)")
        except Exception as e:
            print(f"Seek error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Reset seeking flag after a short delay to allow position update
            self.root.after(300, lambda: setattr(self, 'seeking', False))

    def highlight_segment(self, index):
        """Highlights the specific text segment."""
        try:
            self.transcript_box.tag_remove("highlight", "1.0", tk.END)
            if index < len(self.segments):
                self.transcript_box.tag_add("highlight", f"seg_{index}.first", f"seg_{index}.last")
                self.transcript_box.see(f"seg_{index}.first") # Auto-scroll to text
        except Exception as e:
            # Tag might not exist yet, ignore
            pass

    def update_status(self, text):
        self.status_label.config(text=text)
    
    def toggle_play_pause(self):
        """Toggle between play and pause"""
        if not self.player:
            return
            
        if self.is_playing:
            self.player.pause()
            self.is_playing = False
            self.play_pause_btn.config(text="▶ Play")
        else:
            self.player.play()
            self.is_playing = True
            self.play_pause_btn.config(text="⏸ Pause")
    
    def stop_video(self):
        """Stop video playback"""
        if self.player:
            self.player.stop()
            self.is_playing = False
            self.play_pause_btn.config(text="▶ Play")
            self.time_label.config(text="00:00 / 00:00")
    
    def rewind_10s(self):
        """Rewind 10 seconds"""
        if self.player:
            current_time = self.player.get_time()
            new_time = max(0, current_time - 10000)  # 10 seconds in milliseconds
            self.player.set_time(new_time)
    
    def forward_10s(self):
        """Forward 10 seconds"""
        if self.player:
            current_time = self.player.get_time()
            new_time = current_time + 10000  # 10 seconds in milliseconds
            self.player.set_time(new_time)
    
    def set_volume(self, value):
        """Set volume (0-100)"""
        if self.player:
            volume = int(value)
            self.player.audio_set_volume(volume)
    
    def on_closing(self):
        """Handle window closing - stop video and cleanup"""
        print("Closing application...")
        self.is_playing = False
        if self.player:
            try:
                self.player.stop()
            except:
                pass
        if self.instance:
            try:
                self.instance.release()
            except:
                pass
        self.root.destroy()
        self.root.quit()

def find_video_and_play():
    # Find video files in current directory
    current_dir = os.getcwd()
    
    # Priority order: .mkv files first, then sample2.mp4, then other video files
    video_file = None
    
    # First, look for .mkv files
    mkv_files = [f for f in os.listdir(current_dir) if f.endswith(".mkv")]
    if mkv_files:
        video_file = mkv_files[0]  # Use first .mkv file found
        print(f"Found .mkv file: {video_file}")
    
    # If no .mkv, check for sample2.mp4
    if not video_file and os.path.exists("sample2.mp4"):
        video_file = "sample2.mp4"
        print(f"Found sample2.mp4")
    
    # If still not found, look for any other video file
    if not video_file:
        video_extensions = [".mp4", ".avi", ".mov", ".flv"]
        for ext in video_extensions:
            video_file = next((f for f in os.listdir(current_dir) if f.endswith(ext)), None)
            if video_file:
                print(f"Found video file: {video_file}")
                break

    if not video_file:
        print("No video file found in this folder!")
        print("Looking for: .mkv files, sample2.mp4, or any .mp4/.avi/.mov/.flv file")
        return

    print(f"Loading video: {video_file}")
    root = tk.Tk()
    app = SmartVideoPlayer(root, video_file)
    root.mainloop()

if __name__ == "__main__":
    find_video_and_play()