# Audio Transcription and Summarization Script

**This project was written by Gemini 2.5 Flash LLM by Google.**

A Python script designed to automate the transcription of local audio files with speaker diarization using AssemblyAI, followed by summarization of the transcript using a dynamically selected xAI large language model. The script saves the detailed transcript (with potential speaker names), the generated summary, and process metadata (like audio date/time and xAI usage) to a structured JSON file. It includes robust error handling and retry mechanisms for API calls.

## Features

* **Robust API Calls:** Implements error handling and retry mechanisms with exponential backoff and jitter for AssemblyAI and xAI API interactions, improving reliability against transient issues.
* **Audio Transcription:** Transcribes local MP3 audio files using AssemblyAI's powerful API.
* **Speaker Diarization:** Identifies and labels different speakers in the transcript (e.g., Speaker A, Speaker B).
* **Dynamic LLM Selection:** Fetches available xAI language models and selects a preferred model based on a defined priority pattern (currently prioritizing `grok-N-mini-fast-beta`, `grok-N-beta`, `grok-N` with the highest available `N`, falling back to `grok-3-latest` if available, then the first model in the list).
* **Context-Aware Summarization:** Summarizes the transcript using the selected xAI model with a prompt engineered to:
    * Generate output in **Markdown format**.
    * Use **bullet points** (`- `) for key discussion points.
    * Optionally structure the summary as **meeting minutes** (including date/time and attendees) if the conversation content indicates a meeting.
    * Attempt to **identify speaker names** from the transcript content and use those names instead of generic labels in the summary and the updated transcript. It follows a specific formatting rule for the first mention (e.g., "Bill (Speaker A)" or "Speaker B (name not specified)") and uses just the name or label for subsequent mentions.
* **Metadata Capture:** Records the audio file's modification date/time as a proxy for the recording date.
* **Usage Tracking:** Extracts and saves token usage data from the xAI API response.
* **Structured Output:** Saves the full transcript (with potentially updated speaker names derived from the summary), the Markdown summary, xAI usage info, and audio file information (like modification date/time) to a single, well-structured JSON file (`<audio_file_base_name>_transcription_with_speakers.json`).
* **Idempotency:** Checks for existing output data and skips transcription/summarization by default unless forced, saving time and API costs.
* **Secure API Key Handling:** Supports loading API keys from environment variables (`.env` file) or securely prompting the user via the command line.
* **Configurable Summary Length:** Allows setting the maximum number of tokens for the xAI summary response via a command-line argument.

## Technology Stack

* **Python 3.x:** The core programming language.
* **AssemblyAI Python SDK:** For interacting with the AssemblyAI transcription API.
* **Requests:** For making HTTP requests to the xAI API.
* **Argparse:** For parsing command-line arguments.
* **python-dotenv:** For loading configuration from `.env` files.
* **Getpass:** For securely prompting for API keys.
* **Tenacity:** For implementing retry logic.
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
    tenacity
    ```
    Then install the packages:
    ```bash
    pip install -r requirements.txt
    ```

## Setup

1.  Obtain your AssemblyAI and xAI API keys from their respective provider dashboards.
2.  For convenient and secure API key management, create a file named `.env` in the same directory as the script (`transcribe_audio.py`). Add your keys to this file:
    ```env
    ASSEMBLYAI_API_KEY=your_assemblyai_api_key
    XAI_API_KEY=your_xai_api_key
    ```
    Replace `your_assemblyai_api_key` and `your_xai_api_key` with your actual keys. The script will automatically load these. If environment variables are not set, the script will prompt you to enter the keys securely when it runs.

## Usage

Run the script from your terminal. Navigate to the directory where you saved the script (`transcribe_audio.py`).

```bash
python transcribe_audio.py <audio_file> [options]
```
1. <audio_file> (Required): The name of the local audio file you want to process (e.g., meeting_recording.mp3). The script will look for this file in the current directory and in a ./data/ subdirectory.

2. [options] (Optional):

* -s SPEAKER_COUNT, --speakers SPEAKER_COUNT: Expected number of speakers in the audio. Providing an accurate number can improve AssemblyAI's diarization accuracy. Defaults to 3.
* --force-transcribe: Forces the script to re-run the transcription using AssemblyAI, even if existing transcript data is found in the output file.
* --force-summarize: Forces the script to re-run the summarization using xAI, even if existing summary and usage data are found in the output file.
* --max-tokens MAX_TOKENS: Maximum number of tokens the xAI model should generate for the summary. Adjust this value based on the desired summary length and model limitations. Defaults to 10000.

Examples:

Transcribe and summarize an audio file named conference_call.mp3 with the default settings:
```Bash
python transcribe_audio.py conference_call.mp3
```

Process team_sync.mp3, expecting 5 speakers and requesting a longer summary:
```Bash
python transcribe_audio.py team_sync.mp3 -s 5 --max-tokens 15000
```

Force re-summarization for an audio file you've processed before (old_notes.mp3):
```Bash
python transcribe_audio.py old_notes.mp3 --force-summarize
```

Force both transcription and summarization for a file to get completely fresh output:
```Bash
python transcribe_audio.py important_dialogue.mp3 --force-transcribe --force-summarize
```

## Output
The script generates a JSON file named after the input audio file, with _transcription_with_speakers.json appended (e.g., processing presentation.mp3 will create presentation_transcription_with_speakers.json)
```JSON
{
    "audio_info": {
        "recorded_datetime": "YYYY-MM-DD HH:MM:SS" // File modification date/time
    },
    "transcript": {
        "segments": [
            {
                "speaker": "Speaker A" or "Identified Name", // Original label or name extracted from summary
                "text": "Transcribed text for this segment."
            },
            // ... more segments
        ]
    },
    "summary": "Markdown formatted summary from xAI, potentially in meeting minutes format with bullet points. Speaker names are formatted as 'Name (Speaker X)' or 'Speaker X (name not specified)' on first mention, and by name/label thereafter.",
    "xai_usage": { // Usage data from the xAI API call
        "input_tokens": ...,
        "output_tokens": ...,
        // ... other usage details
    }
}
```
.
