import os
import base64
import io
import tempfile
import threading
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, disconnect
import speech_recognition as sr
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from YeongSil import YeongSil

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'yeongsil_navigation_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
latest_frame = None
is_listening = False
yeongsil_ai = None
processing_queue = []
max_queue_size = 3  # Limit processing queue for performance
audio_processing_lock = threading.Lock()  # Prevent concurrent audio processing
last_processed_audio = 0  # Timestamp of last processed audio to prevent duplicate processing

# Initialize YeongSil AI
try:
    yeongsil_ai = YeongSil()
    print("✅ YeongSil AI initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize YeongSil AI: {e}")
    yeongsil_ai = None

# Initialize speech recognition
recognizer = sr.Recognizer()
print("🎤 Speech recognition initialized")

@app.route('/')
def index():
    """Serve the mobile app HTML page"""
    return render_template('mobile_app.html')

@app.route('/health')
def health():
    """Return server status for connectivity check"""
    return jsonify({
        'status': 'healthy',
        'yeongsil_ready': yeongsil_ai is not None,
        'listening': is_listening
    })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame with YeongSil AI"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name
        
        if yeongsil_ai:
            guidance, depth_buckets = yeongsil_ai.get_guidance(tmp_path)
            os.unlink(tmp_path)  # Clean up temp file
            
            # Convert numpy float32 to regular Python floats for JSON serialization
            depth_buckets_serializable = [float(bucket) for bucket in depth_buckets]
            
            return jsonify({
                'guidance': guidance,
                'depth_buckets': depth_buckets_serializable
            })
        else:
            os.unlink(tmp_path)
            return jsonify({'error': 'YeongSil AI not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("📱 Client connected")
    emit('status', {'message': 'Connected to YeongSil Navigation Assistant'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    global is_listening
    print("📱 Client disconnected")
    is_listening = False

@socketio.on('start_continuous_mode')
def handle_start_continuous():
    """Start continuous listening mode"""
    global is_listening
    
    if not is_listening:
        is_listening = True
        print("🎤 Starting continuous voice recognition...")
        print("📱 Using WebSocket-based audio processing (more reliable)")
        
        emit('status', {'message': 'Voice recognition active - say "scan surroundings"'})

@socketio.on('stop_continuous_mode')
def handle_stop_continuous():
    """Stop continuous listening mode"""
    global is_listening
    is_listening = False
    print("🎤 Stopped continuous voice recognition")
    emit('status', {'message': 'Voice recognition stopped'})

@socketio.on('frame_data')
def handle_frame_data(data):
    """Handle camera frame data"""
    global latest_frame
    
    try:
        # Store the latest frame for voice command processing
        latest_frame = data.get('frame')
        print("📸 Frame received and stored")
    except Exception as e:
        print(f"❌ Error handling frame data: {e}")

@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle audio data for voice recognition with deduplication"""
    global last_processed_audio
    
    try:
        current_time = time.time()
        
        # Prevent processing too many audio chunks in quick succession
        # This helps with the overlapping audio chunks
        if current_time - last_processed_audio < 0.5:  # 500ms minimum between processing
            print(f"🎤 Skipping audio chunk (too recent): {current_time - last_processed_audio:.2f}s ago")
            return
        
        with audio_processing_lock:
            # Decode base64 audio
            audio_data = base64.b64decode(data['audio'])
            audio_format = data.get('format', 'audio/webm')
            
            print(f"🎤 Processing audio data: {len(audio_data)} bytes, format: {audio_format}")
            
            # Determine file extension based on format
            if 'webm' in audio_format:
                suffix = '.webm'
            elif 'mp4' in audio_format:
                suffix = '.mp4'
            elif 'wav' in audio_format:
                suffix = '.wav'
            else:
                suffix = '.webm'  # Default fallback
            
            # Save audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as audio_file:
                audio_file.write(audio_data)
                audio_path = audio_file.name
            
            # Process based on format
            if suffix == '.wav':
                # Direct processing for WAV
                process_voice_command(audio_path)
            else:
                # Convert other formats to WAV
                try:
                    # Try ffmpeg conversion first
                    wav_path = audio_path.replace(suffix, '.wav')
                    import subprocess
                    result = subprocess.run([
                        'ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le', 
                        '-ar', '16000', '-ac', '1', wav_path, '-y'
                    ], capture_output=True, timeout=3)  # Reduced timeout for faster processing
                    
                    if result.returncode == 0:
                        process_voice_command(wav_path)
                        os.unlink(wav_path)
                    else:
                        # Fallback: try pydub conversion
                        process_voice_command_webm(audio_path)
                        
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # Fallback: try pydub conversion
                    process_voice_command_webm(audio_path)
            
            # Clean up
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            
            # Update last processed timestamp
            last_processed_audio = current_time
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        emit('voice_processing_error', {'error': str(e)})

def process_voice_command(audio_path):
    """Process voice command from audio file"""
    try:
        print("🎤 Processing voice command...")
        
        # Use speech recognition
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        
        # Try Google Speech Recognition with improved settings
        try:
            # Configure recognizer for better accuracy
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.8
            
            text = recognizer.recognize_google(audio, language='en-US').lower()
            print(f"🎤 Recognized: '{text}'")
            
            # Check for "scan surroundings" command (more flexible matching)
            if any(phrase in text for phrase in ["scan surroundings", "scan", "scanning"]):
                print("✅ Voice command detected: scan surroundings")
                emit('voice_command_detected', {'command': text})
                
                # Process latest frame if available
                if latest_frame and yeongsil_ai:
                    process_immediate_scan()
                else:
                    emit('voice_analysis_error', {'error': 'No frame available or YeongSil not ready'})
            # Check for "read" command (more flexible matching)
            elif any(phrase in text for phrase in ["read", "reading", "read this", "read that"]):
                print("✅ Voice command detected: read text")
                emit('voice_command_detected', {'command': text})
                
                # Process latest frame for text extraction if available
                if latest_frame and yeongsil_ai:
                    process_text_extraction()
                else:
                    emit('voice_analysis_error', {'error': 'No frame available or YeongSil not ready'})
            else:
                # Only emit voice_detected for longer phrases to reduce noise
                if len(text.split()) >= 2:
                    emit('voice_detected', {'text': text})
                
        except sr.UnknownValueError:
            print("🎤 Could not understand audio")
        except sr.RequestError as e:
            print(f"🎤 Speech recognition error: {e}")
            emit('voice_processing_error', {'error': str(e)})
            
    except Exception as e:
        print(f"❌ Error in voice processing: {e}")
        emit('voice_processing_error', {'error': str(e)})

def process_voice_command_webm(webm_path):
    """Process voice command from WebM audio file"""
    try:
        print("🎤 Processing WebM voice command...")
        
        # Try to use pydub for WebM processing
        try:
            from pydub import AudioSegment
            from pydub.utils import which
            
            # Load WebM file
            audio_segment = AudioSegment.from_file(webm_path, format="webm")
            
            # Convert to WAV format that speech_recognition can handle
            wav_data = audio_segment.export(format="wav").read()
            
            # Save temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_file:
                wav_file.write(wav_data)
                wav_path = wav_file.name
            
            # Process with speech recognition
            process_voice_command(wav_path)
            
            # Clean up
            os.unlink(wav_path)
            
        except ImportError:
            print("❌ pydub not available, trying alternative approach")
            # Fallback: try to process as raw audio
            try:
                with open(webm_path, 'rb') as f:
                    raw_audio = f.read()
                
                # Try to process with speech recognition directly
                # This is a fallback and may not work with all WebM files
                print("⚠️ Attempting direct WebM processing (may not work)")
                
            except Exception as fallback_error:
                print(f"❌ WebM processing failed: {fallback_error}")
                emit('voice_processing_error', {'error': 'WebM audio format not supported'})
                
    except Exception as e:
        print(f"❌ Error in WebM voice processing: {e}")
        emit('voice_processing_error', {'error': str(e)})

# Background processing removed - using WebSocket-based processing instead

def process_immediate_scan():
    """Process immediate scan with latest frame"""
    global processing_queue
    
    try:
        print("🔍 Processing immediate scan...")
        
        # Check processing queue to prevent overload
        if len(processing_queue) >= max_queue_size:
            print("⚠️ Processing queue full, skipping scan")
            emit('voice_analysis_error', {'error': 'Processing queue full, please wait'})
            return
        
        # Add to processing queue
        processing_queue.append(time.time())
        
        # Decode base64 image
        image_data = base64.b64decode(latest_frame.split(',')[1])
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name
        
        # Process with YeongSil
        guidance, depth_buckets = yeongsil_ai.get_guidance(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        # Remove from processing queue
        if processing_queue:
            processing_queue.pop(0)
        
        # Log the guidance for debugging
        print(f"🔍 YeongSil guidance: {guidance}")
        print(f"📊 Depth buckets: {len(depth_buckets)} buckets")
        
        # Convert numpy float32 to regular Python floats for JSON serialization
        depth_buckets_serializable = [float(bucket) for bucket in depth_buckets]
        
        # Send results
        emit('voice_analysis_result', {
            'guidance': guidance,
            'depth_buckets': depth_buckets_serializable
        })
        
        print("✅ Scan processing completed - guidance sent to frontend")
        
    except Exception as e:
        # Remove from processing queue on error
        if processing_queue:
            processing_queue.pop(0)
        print(f"❌ Error in immediate scan: {e}")
        emit('voice_analysis_error', {'error': str(e)})

def process_text_extraction():
    """Process text extraction with latest frame"""
    global processing_queue
    
    try:
        print("📖 Processing text extraction...")
        
        # Check processing queue to prevent overload
        if len(processing_queue) >= max_queue_size:
            print("⚠️ Processing queue full, skipping text extraction")
            emit('voice_analysis_error', {'error': 'Processing queue full, please wait'})
            return
        
        # Add to processing queue
        processing_queue.append(time.time())
        
        # Decode base64 image
        image_data = base64.b64decode(latest_frame.split(',')[1])
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name
        
        # Process with YeongSil text extraction
        extracted_text = yeongsil_ai.get_text_from_image(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        # Remove from processing queue
        if processing_queue:
            processing_queue.pop(0)
        
        # Log the extracted text for debugging
        print(f"📖 Extracted text: {extracted_text}")
        
        # Send results
        emit('voice_analysis_result', {
            'guidance': extracted_text,
            'depth_buckets': []  # No depth data for text extraction
        })
        
        print("✅ Text extraction completed - text sent to frontend")
        
    except Exception as e:
        # Remove from processing queue on error
        if processing_queue:
            processing_queue.pop(0)
        print(f"❌ Error in text extraction: {e}")
        emit('voice_analysis_error', {'error': str(e)})

# Note: Continuous voice processing is now handled via WebSocket audio_data events
# This is more reliable than background threads and avoids Flask context issues

if __name__ == '__main__':
    print("🚀 Starting YeongSil Navigation Assistant...")
    print("📱 Server will be available at http://0.0.0.0:8080")
    print("🎤 Voice recognition: Active")
    print("🤖 YeongSil AI: Ready" if yeongsil_ai else "❌ YeongSil AI: Not available")
    
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)