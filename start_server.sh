#!/bin/bash

echo "🚀 Starting YeongSil Navigation Service..."
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Make sure you're in the hackharvard directory."
    exit 1
fi

# Check if config.py exists and has API key
if [ ! -f "config.py" ]; then
    echo "❌ Error: config.py not found. Please create it with your Gemini API key."
    exit 1
fi

# Get the computer's IP address
echo "📡 Finding your computer's IP address..."
IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
echo "Your computer's IP address: $IP"
echo ""

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt
echo ""

# Start the server
echo "🌐 Starting server..."
echo "Server will be available at: http://$IP:8080"
echo ""
echo "On your phone, open your browser and go to:"
echo "http://$IP:8080"
echo ""
echo "Make sure your phone and computer are on the same WiFi network!"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py
