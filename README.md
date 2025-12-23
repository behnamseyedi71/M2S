# Real-time Video Player with Speech-to-Text Transcription

A Python script that plays video files while simultaneously performing real-time speech-to-text transcription using OpenAI's Whisper model.

## Features

- **Real-time video playback** using OpenCV
- **Live speech-to-text transcription** using faster-whisper (offline, no internet required)
- **Dual-window display**: Video window and transcription window
- **Multi-threaded processing** for smooth playback and transcription
- **Keyboard controls** for pause/resume and quit
- **Automatic audio extraction** from video files

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (required by moviepy for audio extraction)

Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Python Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install opencv-python moviepy faster-whisper torch numpy
```

**Note**: The first time you run the script, faster-whisper will download the Whisper model (automatically). The "base" model is about 150MB. Larger models (small, medium, large) provide better accuracy but are slower.

## Usage

### Basic Usage

If there's a single `.mkv` file in the current directory:

```bash
python video_transcribe.py
```

### Specify Video File

```bash
python video_transcribe.py path/to/your/video.mkv
```

The script supports various video formats (not just .mkv) that moviepy can handle:
- .mkv, .mp4, .avi, .mov, .flv, etc.

## Controls

- **SPACE** - Pause/Resume video playback
- **'q'** or **ESC** - Quit the application

## How It Works

1. **Video Thread**: Main thread handles video playback using OpenCV, displaying frames at the correct frame rate
2. **Transcription Thread**: Background thread extracts audio from the video and processes it in chunks using faster-whisper
3. **Communication**: A queue passes transcribed text from the transcription thread to the video display thread
4. **Display**: Two windows show the video and live transcription text

## Model Selection

You can change the Whisper model size in the script by modifying the `model_size` parameter in the `AudioTranscriber` initialization:

- `"tiny"` - Fastest, least accurate (~39MB)
- `"base"` - Good balance (default, ~150MB)
- `"small"` - Better accuracy (~500MB)
- `"medium"` - High accuracy (~1.5GB)
- `"large"` - Best accuracy (~3GB)

Edit line in `main()`:
```python
transcriber = AudioTranscriber(video_path, transcription_queue, 
                              model_size="base")  # Change here
```

## Performance Tips

1. **GPU Acceleration**: If you have an NVIDIA GPU with CUDA, the script will automatically use it for faster transcription
2. **Model Size**: Use smaller models (tiny/base) for faster processing, larger models for better accuracy
3. **Chunk Duration**: Adjust `chunk_duration` in `transcribe_streaming()` method (default 3 seconds) - smaller chunks = more responsive but more processing overhead

## Troubleshooting

### "No audio track found"
- The video file may not have an audio track
- Try a different video file

### Slow transcription
- Use a smaller model (tiny or base)
- Ensure you have a GPU if available
- Reduce chunk_duration for faster updates (but may reduce accuracy)

### Video playback issues
- Ensure OpenCV can read your video format
- Check that the video file is not corrupted

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Ensure FFmpeg is installed and accessible in your PATH

## License

This script is provided as-is for educational and personal use.

