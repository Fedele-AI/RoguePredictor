#!/bin/bash

# Rouge Wave Analysis Installation Script
# Installs dependencies and sets up the environment

echo "=========================================="
echo "ROUGE WAVE ANALYSIS INSTALLATION"
echo "IBM-NASA Geospatial Models Integration"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "✓ Python version: $PYTHON_VERSION"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "✓ pip3 is available"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created successfully"
else
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -eq 0 ]; then
    echo "✓ Virtual environment activated"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    echo "You may need to install some system dependencies first."
    echo "On Ubuntu/Debian: sudo apt-get install python3-dev build-essential"
    echo "On macOS: xcode-select --install"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data
mkdir -p outputs
mkdir -p logs
mkdir -p model_cache
echo "✓ Project directories created"

# Test the installation
echo ""
echo "Testing installation..."
python test_system.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 INSTALLATION COMPLETED SUCCESSFULLY!"
    echo ""
    echo "To get started:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run the demo: python demo.py"
    echo "3. Run the full analysis: python rouge_wave_analysis.py"
    echo "4. Check help: python rouge_wave_analysis.py --help"
    echo ""
    echo "Happy analyzing! 🌊"
else
    echo ""
    echo "⚠️  Installation completed but some tests failed."
    echo "The system may still work, but check the test output above."
fi

# Deactivate virtual environment
deactivate

echo ""
echo "Installation script completed." 