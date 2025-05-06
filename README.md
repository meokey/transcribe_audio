# Audio Transcription and Summarization Script

A Python script designed to automate the transcription of local audio files with speaker diarization using AssemblyAI, followed by summarization of the transcript using a dynamically selected xAI large language model. The script saves the detailed transcript (with potential speaker names), the generated summary, and process metadata to a structured JSON file.

## Features

* **Audio Transcription:** Transcribes local MP3 audio files using AssemblyAI's powerful API.
* **Speaker Diarization:** Identifies and labels different speakers in the transcript (e.g., Speaker A, Speaker B).
* **Dynamic LLM Selection:** Fetches available xAI language models and selects a preferred model based on a defined priority pattern (currently prioritizing recent Grok models) for summarization.
* **Context-Aware Summarization:** Summarizes the transcript using the selected xAI model with a prompt engineered to:
    * Generate output in **Markdown format**.
    * Use **bullet points** for key discussion points.
    * Optionally structure the summary as **meeting minutes** (including date/time and attendees) if the conversation content indicates a meeting.
    * Attempt to **identify speaker names** from the transcript content and use those names instead of generic labels in the summary and the updated transcript.
* **Metadata Capture:** Records the audio file's modification date/time as a proxy for the recording date.
* **Usage Tracking:** Extracts and saves token usage data from the xAI API response.
* **Structured Output:** Saves the full transcript (with updated speaker names), the Markdown summary, xAI usage info, and audio metadata to a single, well-structured JSON file (`<audio_file_base_name>_transcription_with_speakers.json`).
* **Idempotency:** Checks for existing output data and skips transcription/summarization by default unless forced, saving time and API costs.
* **Secure API Key Handling:** Supports loading API keys from environment variables (`.env` file) or securely prompting the user via the command line.

## Technology Stack

* **Python 3.x:** The core programming language.
* **AssemblyAI Python SDK:** For interacting with the AssemblyAI transcription API.
* **Requests:** For making HTTP requests to the xAI API.
* **Argparse:** For parsing command-line arguments.
* **python-dotenv:** For loading configuration from `.env` files.
* **Getpass:** For securely prompting for API keys.
* **Standard Python Libraries:** `os`, `json`, `re`, `datetime`.

## Prerequisites

* Python 3.x installed on your system.
* An active AssemblyAI account and API key.
* Access to the xAI API and an API key.

## Installation

1.  Clone the repository (if the script is part of a repo):
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  Create a virtual environment (highly recommended) and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: `.venv\Scripts\activate`
    ```
3.  Install the required Python packages. Create a `requirements.txt` file in the script's directory with the following content:
    ```
    assemblyai
    requests
    python-dotenv
    ```
    Then install the packages:
    ```bash
    pip install -r requirements.txt
    ```

## Setup

1.  Obtain your AssemblyAI and xAI API keys from their respective provider dashboards.
2.  For convenient and secure API key management, create a file named `.env` in the same directory as the script (`process_audio.py`). Add your keys to this file:
    ```env
    ASSEMBLYAI_API_KEY=your_assemblyai_api_key
    XAI_API_KEY=your_xai_api_key
    ```
    Replace `your_assemblyai_api_key` and `your_xai_api_key` with your actual keys. The script will automatically load these. If environment variables are not set, the script will prompt you to enter the keys securely when it runs.

## Usage

Run the script from your terminal. Navigate to the directory where you saved the script (`process_audio.py`).

```bash
python process_audio.py <audio_file> [options]

1. <audio_file> (Required): The name of the local audio file you want to process (e.g., meeting_recording.mp3). The script will look for this file in the current directory and in a ./data/ subdirectory.

2. [options] (Optional):

- -s SPEAKER_COUNT, --speakers SPEAKER_COUNT: Expected number of speakers in the audio. Providing an accurate number can improve AssemblyAI's diarization accuracy. Defaults to 3.
- --force-transcribe: Forces the script to re-run the transcription using AssemblyAI, even if existing transcript data is found in the output file.
- --force-summarize: Forces the script to re-run the summarization using xAI, even if existing summary and usage data are found in the output file.
