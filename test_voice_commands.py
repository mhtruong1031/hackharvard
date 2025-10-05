#!/usr/bin/env python3
"""
Test script for YeongSil voice command functionality
This script tests the voice recognition and image processing pipeline
"""

import os
import sys
import time
import requests
import base64
from pathlib import Path

def test_server_health():
    """Test if the server is running and healthy"""
    try:
        response = requests.get('http://localhost:8080/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy")
            print(f"   - Status: {data.get('status')}")
            print(f"   - YeongSil Ready: {data.get('yeongsil_ready')}")
            print(f"   - Listening: {data.get('listening')}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_image_processing():
    """Test image processing with a sample image"""
    try:
        # Check if we have any test images
        test_images = list(Path('.').glob('temp_continuous_*.jpg'))
        if not test_images:
            print("⚠️ No test images found, skipping image processing test")
            return True
        
        # Use the most recent test image
        test_image = max(test_images, key=os.path.getctime)
        print(f"📸 Testing with image: {test_image}")
        
        # Read and encode image
        with open(test_image, 'rb') as f:
            image_data = f.read()
        
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Test the process_frame endpoint
        response = requests.post(
            'http://localhost:8080/process_frame',
            json={'image': f'data:image/jpeg;base64,{base64_image}'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Image processing successful")
            print(f"   - Guidance: {data.get('guidance', 'No guidance')[:100]}...")
            print(f"   - Depth buckets: {len(data.get('depth_buckets', []))} buckets")
            return True
        else:
            print(f"❌ Image processing failed: {response.status_code}")
            print(f"   - Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing image processing: {e}")
        return False

def test_voice_commands():
    """Test voice command recognition"""
    print("🎤 Voice command testing requires manual interaction")
    print("   - Start the server: python app.py")
    print("   - Open http://localhost:8080 in a mobile browser")
    print("   - Click 'Start Navigation Service'")
    print("   - Say 'scan surroundings' and verify:")
    print("     * Voice command is detected")
    print("     * Image is processed")
    print("     * Guidance is spoken back")
    return True

def main():
    """Run all tests"""
    print("🧪 YeongSil Voice Command Test Suite")
    print("=" * 50)
    
    # Test 1: Server Health
    print("\n1. Testing server health...")
    if not test_server_health():
        print("❌ Server is not running. Please start with: python app.py")
        return False
    
    # Test 2: Image Processing
    print("\n2. Testing image processing...")
    if not test_image_processing():
        print("❌ Image processing test failed")
        return False
    
    # Test 3: Voice Commands (Manual)
    print("\n3. Testing voice commands...")
    test_voice_commands()
    
    print("\n✅ All automated tests passed!")
    print("\n📱 Manual Testing Instructions:")
    print("   1. Ensure server is running: python app.py")
    print("   2. Open http://localhost:8080 on mobile device")
    print("   3. Allow camera and microphone permissions")
    print("   4. Click 'Start Navigation Service'")
    print("   5. Say 'scan surroundings' clearly")
    print("   6. Verify guidance is spoken back")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
