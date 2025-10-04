from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
import time
import threading
import queue
import speech_recognition as sr
import pyaudio
import wave
from YeongSil import YeongSil

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSocket with CORS

# Initialize the YeongSil system
yeong_sil = YeongSil()

# Global variables for continuous processing
frame_queue = queue.Queue(maxsize=5)
audio_queue = queue.Queue(maxsize=10)
is_listening = False
is_processing_continuous = False
pending_scan_request = False  # Flag to indicate if we should process the next frame

@app.route('/')
def landing():
    return render_template('mobile_app.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "YeongSil server is running"})

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to YeongSil server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    global is_listening, is_processing_continuous
    is_listening = False
    is_processing_continuous = False

@socketio.on('start_continuous_mode')
def handle_start_continuous():
    """Start continuous camera and audio processing"""
    global is_processing_continuous, is_listening
    is_processing_continuous = True
    is_listening = True
    emit('status', {'message': 'Continuous mode started - always listening for "scan surroundings"'})
    
    # Start background processing threads
    threading.Thread(target=continuous_frame_processor, daemon=True).start()
    threading.Thread(target=continuous_audio_processor, daemon=True).start()

@socketio.on('stop_continuous_mode')
def handle_stop_continuous():
    """Stop continuous processing"""
    global is_processing_continuous, is_listening
    is_processing_continuous = False
    is_listening = False
    emit('status', {'message': 'Continuous mode stopped'})

@socketio.on('frame_data')
def handle_frame_data(data):
    """Receive frame data from client"""
    if is_processing_continuous:
        try:
            frame_queue.put_nowait(data)
        except queue.Full:
            # Remove oldest frame if queue is full
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(data)
            except queue.Empty:
                pass

@socketio.on('audio_data')
def handle_audio_data(data):
    """Receive audio data from client"""
    if is_listening:
        try:
            audio_queue.put_nowait(data)
        except queue.Full:
            # Remove oldest audio if queue is full
            try:
                audio_queue.get_nowait()
                audio_queue.put_nowait(data)
            except queue.Empty:
                pass

def continuous_frame_processor():
    """Background thread for processing continuous video frames"""
    global is_processing_continuous, pending_scan_request
    
    while is_processing_continuous:
        try:
            if not frame_queue.empty() and pending_scan_request:
                frame_data = frame_queue.get(timeout=1)
                pending_scan_request = False  # Reset the flag
                
                # Process frame with YeongSil
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(frame_data['frame'])
                    image = Image.open(io.BytesIO(image_data))
                    image_array = np.array(image)
                    
                    # Save temporary image
                    timestamp = frame_data.get('timestamp', int(time.time()))
                    temp_path = f"temp_continuous_{timestamp}.jpg"
                    cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
                    
                    # Process with YeongSil
                    guidance, depth_buckets = yeong_sil.get_guidance(temp_path)
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                    # Send result via WebSocket
                    socketio.emit('analysis_result', {
                        'guidance': guidance,
                        "depth_buckets": [float(x) for x in depth_buckets],
                        'timestamp': timestamp,
                        'type': 'continuous_scan'
                    })
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Continuous frame processor error: {e}")
            time.sleep(1)

def continuous_audio_processor():
    """Background thread for processing continuous audio"""
    global is_listening, pending_scan_request
    
    # Initialize speech recognition
    r = sr.Recognizer()
    
    while is_listening:
        try:
            if not audio_queue.empty():
                audio_data = audio_queue.get(timeout=1)
                
                # Process audio for voice commands
                try:
                    # Convert base64 audio to audio data
                    audio_bytes = base64.b64decode(audio_data['audio'])
                    
                    # Save temporary audio file
                    temp_audio_path = f"temp_audio_{int(time.time())}.wav"
                    with open(temp_audio_path, 'wb') as f:
                        f.write(audio_bytes)
                    
                    # Use speech recognition
                    with sr.AudioFile(temp_audio_path) as source:
                        audio = r.record(source)
                    
                    # Try to recognize speech
                    try:
                        text = r.recognize_google(audio).lower()
                        print(f"Heard: {text}")
                        
                        # Check for "scan surroundings" command
                        if 'scan' in text and 'surroundings' in text:
                            print("🎤 VOICE COMMAND DETECTED: 'scan surroundings' - Processing navigation request!")
                            pending_scan_request = True  # Set flag to process next frame
                            socketio.emit('voice_command_detected', {
                                'command': 'scan_surroundings',
                                'text': text,
                                'timestamp': time.time()
                            })
                                
                    except sr.UnknownValueError:
                        # Speech not recognized, continue listening
                        pass
                    except sr.RequestError as e:
                        print(f"Speech recognition error: {e}")
                    
                    # Clean up temp file
                    os.remove(temp_audio_path)
                    
                except Exception as e:
                    print(f"Audio processing error: {e}")
                    
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Continuous audio processor error: {e}")
            time.sleep(1)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame when requested"""
    try:
        data = request.json
        frame_data = data.get('frame')  # Base64 encoded image
        timestamp = data.get('timestamp', int(time.time()))
        
        if not frame_data:
            return jsonify({"error": "No frame data provided"}), 400
        
        print(f"Processing frame {timestamp}...")
        
        # Decode base64 image
        image_data = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # Save temporary image
        temp_path = f"temp_frame_{timestamp}.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        
        # Process with YeongSil
        guidance, depth_buckets = yeong_sil.get_guidance(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        print(f"Processing complete for frame {timestamp}")
        
        return jsonify({
            "guidance": guidance,
            "depth_buckets": [float(x) for x in depth_buckets],
            "timestamp": timestamp,
            "status": "completed"
        })
        
    except Exception as e:
        print(f"Processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/voice_command', methods=['POST'])
def voice_command():
    """Handle voice commands from mobile app"""
    try:
        data = request.json
        command = data.get('command', '').lower()
        
        if 'scan' in command or 'analyze' in command:
            return jsonify({
                "response": "Please point your camera at the area you want to analyze and tap the scan button.",
                "action": "request_frame"
            })
        elif 'help' in command:
            return jsonify({
                "response": "You can say 'scan surroundings' to analyze your environment, or 'help' for more options.",
                "action": "none"
            })
        else:
            return jsonify({
                "response": "I didn't understand that command. Try saying 'scan surroundings'.",
                "action": "none"
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting YeongSil Mobile Web App with Continuous Processing...")
    print("Make sure your phone and computer are on the same WiFi network")
    print("Server will be available at: http://YOUR_COMPUTER_IP:8080")
    print("Features: Continuous camera feed, always-listening voice commands")
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)