#!/bin/bash

# Script to set up a basic Git repository for the audio processing project

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Initializing Git repository..."
git init

# --- Optional: Set up Git LFS for large audio files ---
# If you plan to store the original audio files in this repository,
# Git LFS (Large File Storage) is highly recommended.
# Make sure Git LFS is installed on your system before running 'git lfs install'.
# You might need to uncomment the line below and run it once if you've never used Git LFS before.
# echo "Installing Git LFS (if not already installed)..."
# git lfs install

echo "Setting up Git LFS to track audio files..."
# Tracks MP3 files. Adjust the pattern if you handle other audio formats.
git lfs track "*.mp3"

echo "Creating .gitignore file..."
# Define files and directories to ignore
cat <<EOF > .gitignore
# Ignore environment variables file containing secrets
.env

# Ignore Python cache and environment directories
__pycache__/
.venv/
venv/
*.pyc
*.pyo
*.pyd

# Ignore the generated output JSON files
*_transcription_with_speakers.json

# Ignore the temporary AssemblyAI upload manifest (if used by SDK internally)
.assemblyai/

# Ignore IDE files
.idea/
.vscode/

# Ignore OS generated files
.DS_Store
Thumbs.db
EOF

echo "Creating .env.template file..."
# Create a template file for environment variables
cat <<EOF > .env.template
# Environment variables for API keys
# Rename this file to .env and replace the placeholder values with your actual keys

# Get your key from https://www.assemblyai.com/
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here

# Get your key from xAI's API dashboard
XAI_API_KEY=your_xai_api_key_here
EOF

echo "Adding project files to Git..."
# Assuming your main script file is named 'process_audio.py'
# Add the script, requirements file, gitignore, and the env template
git add process_audio.py requirements.txt .gitignore .env.template

# Add the .gitattributes file which is created by 'git lfs track'
git add .gitattributes

echo "Making initial commit..."
git commit -m "Initial commit: Add audio processing script, dependencies, and git setup files"

echo ""
echo "-----------------------------------------------------"
echo "Basic Git repository setup complete."
echo "You can now add your remote origin and push your code:"
echo ""
echo "  git remote add origin <your_repository_url>"
echo "  git push -u origin main" # Assuming 'main' is your default branch name
echo ""
echo "Remember to install Git LFS if you haven't already ('git lfs install') and make sure it's working if you are tracking audio files."
echo "Create your .env file from .env.template and add your actual API keys."
echo "-----------------------------------------------------"
