# Audio Transcription and Summarization Script

**This project was written by Gemini 2.5 Flash LLM by Google.**

A Python script designed to automate the transcription of local audio files with speaker diarization using AssemblyAI, followed by summarization of the transcript using a dynamically selected xAI large language model. The script saves the detailed transcript (with potential speaker names), the generated summary, and process metadata (like audio date/time and xAI usage) to a structured JSON file. It includes robust error handling and retry mechanisms for API calls. Recent enhancements have further improved its processing speed for batch tasks and overall operational robustness.

## License

This project is licensed under the **Polyform Noncommercial License 1.0.0**.

This means you are free to use, study, modify, and distribute the software for **personal, noncommercial purposes**.

**Commercial use,** which includes gaining commercial advantage or monetary compensation from the software or services using the software, is **restricted** by this license.

Please see the full [LICENSE](LICENSE) file for details.

## Features

* **Input Source Flexibility:** Process a single local file, a remote file downloaded from a URL, or all supported audio/video files found within a specified folder and its subdirectories.
* **Expanded Supported Formats:** Supports a wide range of audio and video file formats compatible with AssemblyAI (including common formats like MP3, WAV, MP4, MOV, etc.).
* **Robust API Calls:** Implements error handling and retry mechanisms with exponential backoff and jitter for AssemblyAI and xAI API interactions, improving reliability against transient issues.
* **Audio/Video Transcription:** Transcribes supported audio and video files using AssemblyAI's powerful API.
* **Speaker Diarization:** Identifies and labels different speakers in the transcript (e.g., Speaker A, Speaker B).
* **Dynamic LLM Selection:** Fetches available xAI language models and selects a preferred model based on a defined priority pattern (currently prioritizing `grok-N-mini-fast-beta`, `grok-N-beta`, `grok-N` with the highest available `N`, falling back to `grok-3-latest` if available, then the first model in the list).
* **Context-Aware Summarization:** Summarizes the transcript using the selected xAI model with a prompt engineered to:
    * Generate output in **Markdown format**.
    * Use **bullet points** (`- `) for key discussion points.
    * Optionally structure the summary as **meeting minutes** (including date/time and attendees) if the conversation content indicates a meeting.
    * Attempt to **identify speaker names** from the transcript content and use those names instead of generic labels in the summary and the updated transcript. It follows a specific formatting rule for the first mention (e.g., "Bill (Speaker A)" or "Speaker B (name not specified)") and uses just the name or label for subsequent mentions.
* **Visual Progress Bar:** Provides a `tqdm` progress bar for real-time visual feedback. When processing multiple files in a folder, it displays overall progress based on audio duration (in seconds).
* **Local Duration Fallback:** Utilizes locally detected audio duration (via `mutagen` or `tinytag`) for the overall progress bar update if AssemblyAI does not provide a duration for a specific file.
* **Metadata Capture:** Records the audio file's modification date/time as a proxy for the recording date.
* **Usage Tracking:** Extracts and saves token usage data from the xAI API response.
* **Structured Output:** For each input file, saves the full transcript (with potentially updated speaker names derived from the summary), the Markdown summary, xAI usage info, and audio file information (like original path, modification date/time, and duration) to a single, well-structured JSON file (`<audio_file_base_name>_transcription_with_speakers.json`).
* **Idempotency:** Checks for existing output data and skips transcription/summarization by default unless forced, saving time and API costs.
* **Secure API Key Handling:** Supports loading API keys from environment variables (`.env` file) or securely prompting the user via the command line.
* **Configurable Summary Length:** Allows setting the maximum number of tokens for the xAI summary response via a command-line argument.
* **Concurrent Folder Processing:** Processes multiple files within a folder concurrently (using a configurable number of worker threads via the `--num-workers` argument) to significantly speed up batch operations.
* **Enhanced Robustness & Efficiency:** Improved internal handling of optional dependency installations (using `subprocess` instead of `os.system`), more robust file download logic (including Content-Disposition handling and better fallback mechanisms), and optimized API key/model fetching to reduce redundant calls.
* **Verbose Logging:** Use the `--verbose` flag for detailed, step-by-step output during processing.

## Technology Stack

* **Python 3.x:** The core programming language.
* **AssemblyAI Python SDK:** For interacting with the AssemblyAI transcription API.
* **Requests:** For making HTTP requests to the xAI API.
* **Argparse:** For parsing command-line arguments.
* **python-dotenv:** For loading configuration from `.env` files.
* **Getpass:** For securely prompting for API keys.
* **Tenacity:** For implementing retry logic.
* **Tqdm:** For displaying progress bars.
* **Mutagen:** (Optional, auto-installed) For local audio/video metadata detection, including duration.
* **Tinytag:** (Optional, auto-installed) Another library for local audio metadata detection, including duration.
* **Standard Python Libraries:** `os`, `json`, `re`, `datetime`, `time`, `math`, `logging`, `sys`.

## Prerequisites

* Python 3.x installed on your system.
* An active AssemblyAI account and API key.
* Access to the xAI API and an API key.
* Internet connection to access AssemblyAI and xAI APIs, and for downloading files from URLs.

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
3.  Install the required Python packages.
    Install the packages in Production environment:
    ```bash
    pip install -r requirements.txt
    ```
    In the development environment, install the packages using
    ```bash
    pip install -e .[dev]
    ```
    *(Note: The script will attempt to install `tqdm`, `mutagen`, and `tinytag` automatically if they are not found, but installing dependencies via `pip` in a virtual environment is the recommended practice.)*

## Setup

1.  Obtain your AssemblyAI and xAI API keys from their respective provider dashboards.
2.  For convenient and secure API key management, create a file named `.env` in the same directory as the script. Add your keys to this file:
    ```env
    ASSEMBLYAI_API_KEY=your_assemblyai_api_key
    XAI_API_KEY=your_xai_api_key
    ```
    Replace `your_assemblyai_api_key` and `your_xai_api_key` with your actual keys. The script will automatically load these. If environment variables are not set, the script will prompt you to enter the keys securely when it runs.

## Usage

Run the script from your terminal. Navigate to the directory where you saved the script (`src/transcribe_audio_app/transcribe_audio.py`).

```bash
python src/transcribe_audio_app/transcribe_audio.py <audio_file> [options]
```
1. <input_source> (Required): The path to a local audio/video file, a URL pointing to a remote audio/video file, or the path to a folder containing audio/video files you want to process.

2. [options] (Optional):

* -s SPEAKER_COUNT, --speakers SPEAKER_COUNT: Expected number of speakers in the audio. Providing an accurate number can improve AssemblyAI's diarization accuracy. Defaults to 3.
* --num-workers NUM_WORKERS: Number of concurrent worker threads for processing files when a folder is provided as input. Defaults to 3.
* --force-transcribe: Forces the script to re-run the transcription using AssemblyAI, even if existing transcript data is found in the output file.
* --force-summarize: Forces the script to re-run the summarization using xAI, even if existing summary and usage data are found in the output file.
* --max-tokens MAX_TOKENS: Maximum number of tokens the xAI model should generate for the summary. Adjust this value based on the desired summary length and model limitations. Defaults to 10000.
* --verbose: Enable verbose logging for detailed step-by-step output, including debug information and full API responses. Disables the tqdm progress bar.

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
During folder processing in non-verbose mode, a tqdm progress bar will show the overall progress based on the total duration of supported audio files. Per-file summary details (like date/time, duration, speakers, xAI tokens, and output path) will be printed above the progress bar as each file completes. In verbose mode, detailed logs will be printed instead of the progress bar.
