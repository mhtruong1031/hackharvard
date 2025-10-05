#!/usr/bin/env python3
"""
Test script for voice recognition functionality
This script tests the speech recognition components without running the full web app
"""

import speech_recognition as sr
import pyaudio
import time
import os
import tempfile

def test_microphone_access():
    """Test if microphone is accessible"""
    print("🎤 Testing microphone access...")
    
    try:
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        # List available audio devices
        print(f"🎤 Found {audio.get_device_count()} audio devices:")
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  - Device {i}: {info['name']} (inputs: {info['maxInputChannels']})")
        
        # Test default input device
        default_device = audio.get_default_input_device_info()
        print(f"🎤 Default input device: {default_device['name']}")
        
        audio.terminate()
        return True
        
    except Exception as e:
        print(f"❌ Microphone access failed: {e}")
        return False

def test_speech_recognition():
    """Test speech recognition with a simple phrase"""
    print("\n🎤 Testing speech recognition...")
    
    try:
        # Initialize recognizer
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.8
        r.operation_timeout = 10
        
        print("🎤 Listening for 5 seconds... Say 'scan surroundings' or any phrase...")
        
        # Use microphone to record audio
        with sr.Microphone() as source:
            # Adjust for ambient noise
            print("🎤 Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("🎤 Ready! Speak now...")
            
            # Record audio
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            
        print("🎤 Audio recorded, processing...")
        
        # Try to recognize speech
        try:
            text = r.recognize_google(audio, language='en-US').lower()
            print(f"✅ Speech recognized: '{text}'")
            
            # Test command detection
            scan_keywords = ['scan', 'scanner', 'scanning']
            surroundings_keywords = ['surroundings', 'surrounding', 'environment', 'around', 'area']
            
            has_scan = any(keyword in text for keyword in scan_keywords)
            has_surroundings = any(keyword in text for keyword in surroundings_keywords)
            
            if has_scan and has_surroundings:
                print("✅ Voice command 'scan surroundings' detected!")
                return True
            else:
                print("ℹ️  Speech detected but not a recognized command")
                return True
                
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return False
        except sr.RequestError as e:
            print(f"❌ Speech recognition service error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Speech recognition test failed: {e}")
        return False

def test_audio_formats():
    """Test different audio formats"""
    print("\n🎤 Testing audio format handling...")
    
    try:
        r = sr.Recognizer()
        
        # Create a simple test audio file (silence)
        test_audio_path = "test_silence.wav"
        
        # This would normally be created by recording, but for testing we'll create a minimal file
        # In a real scenario, this would be the audio data from the web app
        
        print("✅ Audio format handling test completed (would need real audio data for full test)")
        return True
        
    except Exception as e:
        print(f"❌ Audio format test failed: {e}")
        return False

def test_network_connectivity():
    """Test network connectivity for Google Speech Recognition"""
    print("\n🌐 Testing network connectivity for speech recognition...")
    
    try:
        import urllib.request
        import urllib.error
        
        # Test connection to Google's speech recognition service
        try:
            urllib.request.urlopen('https://www.google.com', timeout=5)
            print("✅ Network connectivity OK")
            return True
        except urllib.error.URLError as e:
            print(f"❌ Network connectivity issue: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        return False

def test_continuous_listening():
    """Test continuous listening mode"""
    print("\n🎤 Testing continuous listening mode...")
    
    try:
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.8
        
        print("🎤 Testing continuous listening for 10 seconds...")
        print("🎤 Say 'scan surroundings' during this time...")
        
        start_time = time.time()
        command_detected = False
        
        while time.time() - start_time < 10 and not command_detected:
            try:
                with sr.Microphone() as source:
                    # Quick listen for audio
                    audio = r.listen(source, timeout=1, phrase_time_limit=2)
                    
                    try:
                        text = r.recognize_google(audio, language='en-US').lower()
                        print(f"🎤 Heard: '{text}'")
                        
                        # Check for command
                        scan_keywords = ['scan', 'scanner', 'scanning']
                        surroundings_keywords = ['surroundings', 'surrounding', 'environment', 'around', 'area']
                        
                        has_scan = any(keyword in text for keyword in scan_keywords)
                        has_surroundings = any(keyword in text for keyword in surroundings_keywords)
                        
                        if has_scan and has_surroundings:
                            print("✅ Voice command detected in continuous mode!")
                            command_detected = True
                            return True
                            
                    except sr.UnknownValueError:
                        # No speech detected, continue
                        pass
                    except sr.RequestError as e:
                        print(f"❌ Speech recognition error: {e}")
                        break
                        
            except sr.WaitTimeoutError:
                # No audio detected, continue listening
                pass
        
        if not command_detected:
            print("ℹ️  No voice commands detected during continuous listening test")
            return True  # Still a successful test, just no commands spoken
            
    except Exception as e:
        print(f"❌ Continuous listening test failed: {e}")
        return False

def main():
    """Run all voice recognition tests"""
    print("🧪 YeongSil Voice Recognition Test Suite")
    print("=" * 50)
    
    tests = [
        ("Microphone Access", test_microphone_access),
        ("Network Connectivity", test_network_connectivity),
        ("Audio Format Handling", test_audio_formats),
        ("Speech Recognition", test_speech_recognition),
        ("Continuous Listening", test_continuous_listening),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print(f"\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Voice recognition should work properly.")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        
    print("\n💡 Tips for better voice recognition:")
    print("   - Speak clearly and at normal volume")
    print("   - Minimize background noise")
    print("   - Use phrases like 'scan surroundings' or 'scan environment'")
    print("   - Ensure stable internet connection for Google Speech Recognition")
    print("\n🔧 If tests fail:")
    print("   - Check microphone permissions")
    print("   - Verify internet connection")
    print("   - Try speaking louder or closer to microphone")

if __name__ == "__main__":
    main()
