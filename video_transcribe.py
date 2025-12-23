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
from tkinter import ttk
import vlc  # Import vlc AFTER setting the environment variable
import whisper

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

    def populate_text(self):
        """Fills the text box with the transcribed text."""
        self.update_transcript_display()

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