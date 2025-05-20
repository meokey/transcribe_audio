import os
import assemblyai
import json
import argparse
from dotenv import load_dotenv
import getpass
import requests
import re
import datetime
import tenacity
import logging
import sys
import time
import math
import subprocess
import concurrent.futures

# Check for tqdm and install if necessary
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "tqdm"], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        from tqdm import tqdm
        print("tqdm installed successfully.")
    else:
        print(f"WARNING: tqdm installation failed. Progress bars will not be available. Error: {result.stderr}", file=sys.stderr)
        # Define a dummy tqdm if installation fails so the script doesn't crash
        def tqdm(iterable, *args, **kwargs):
            return iterable

# Check for mutagen and install if necessary
try:
    from mutagen import File as MutagenFile
except ImportError:
    print("Installing mutagen for audio/video metadata detection (attempt 1/2)...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "mutagen"], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        try:
            from mutagen import File as MutagenFile
            print("Mutagen installed successfully.")
        except ImportError:
            MutagenFile = None
            print("WARNING: Mutagen installed but could not be imported. Duration detection might be affected.", file=sys.stderr)
    else:
        MutagenFile = None
        print(f"WARNING: Mutagen installation failed. Duration detection might be affected. Error: {result.stderr}", file=sys.stderr)

# Check for tinytag and install if necessary
try:
    from tinytag import TinyTag
except ImportError:
    print("Installing tinytag for audio metadata detection (attempt 2/2)...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "tinytag"], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        try:
            from tinytag import TinyTag
            print("Tinytag installed successfully.")
        except ImportError:
            TinyTag = None
            print("WARNING: Tinytag installed but could not be imported. Duration detection might be affected.", file=sys.stderr)
    else:
        TinyTag = None
        print(f"WARNING: Tinytag installation failed. Duration detection might be affected. Error: {result.stderr}", file=sys.stderr)


# --- Global Variables for Caching ---
_ASSEMBLYAI_API_KEY = None
_XAI_API_KEY = None
_SELECTED_XAI_MODEL = None

# --- Configuration ---
XAI_API_BASE_URL = "https://api.x.ai"
XAI_MESSAGES_ENDPOINT = f"{XAI_API_BASE_URL}/v1/messages"
XAI_MODELS_ENDPOINT = f"{XAI_API_BASE_URL}/v1/models"

TRANSCRIPT_FILENAME_SUFFIX = "_transcription_with_speakers.json"
DOWNLOAD_FOLDER = "./downloads/"

# List of file extensions supported by AssemblyAI (based on documentation link)
SUPPORTED_EXTENSIONS = {
    '.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a', '.wma', '.aiff', '.aif', '.opus',
    '.mp4', '.mov', '.wmv', '.avi', '.mkv', '.webm', '.flv', '.ogv', '.3gp', '.3g2'
}

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) # Default level to only show Warning/Error

# Create formatters
formatter_verbose = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
formatter_non_verbose = logging.Formatter('%(message)s') # Message only

# Create a simplified custom handler for WARNING/ERROR that can use tqdm.write
class TqdmWarningErrorHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET, overall_pbar=None):
        super().__init__(level)
        self.overall_pbar = overall_pbar

    def emit(self, record):
        # Only handle WARNING and ERROR levels
        if record.levelno < logging.WARNING:
            return # Ignore INFO, DEBUG

        try:
            msg = self.format(record)
            # Use tqdm.write for WARNING/ERROR if a bar is active
            if self.overall_pbar and not self.overall_pbar.disable:
                 self.overall_pbar.write(msg)
                 self.overall_pbar.refresh() # Ensure bar is redrawn after write
            else:
                 # Fallback to print for WARNING/ERROR if no bar or bar disabled
                 print(msg, file=sys.stderr, flush=True) # Print errors to stderr


        except Exception:
            self.handleError(record)


# Replace default handler with our custom one
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)

# This handler will only be active for WARNING and ERROR levels in non-verbose
# In verbose mode, we will replace this with a standard handler
warning_error_handler = TqdmWarningErrorHandler(level=logging.WARNING)
warning_error_handler.setFormatter(formatter_non_verbose) # Default to non-verbose formatter
logger.addHandler(warning_error_handler)

# Configure requests and urllib3 loggers to use the same handler for WARNING/ERROR
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
for handler in logging.getLogger("requests").handlers: logging.getLogger("requests").removeHandler(handler)
for handler in logging.getLogger("urllib3").handlers: logging.getLogger("urllib3").removeHandler(handler)
logging.getLogger("requests").addHandler(warning_error_handler)
logging.getLogger("urllib3").addHandler(warning_error_handler)


# --- Retry Configuration ---
# Use logger.warning for retry messages, will be handled by TqdmWarningErrorHandler
common_retry_settings = tenacity.retry(
    wait=tenacity.wait_exponential_jitter(
        max=60, jitter=10
    ),
    stop=tenacity.stop_after_attempt(5),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING), # Log retries at WARNING
    reraise=True,
)

retry_if_transient_http_error = tenacity.retry_if_exception(
    lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code in {429, 500, 502, 503, 504}
)

assemblyai_retry_settings = tenacity.retry(
    wait=tenacity.wait_exponential_jitter(
        max=60, jitter=10
    ),
    stop=tenacity.stop_after_attempt(5),
    retry=(
        tenacity.retry_if_exception_type(requests.exceptions.RequestException)
        | retry_if_transient_http_error
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING), # Log retries at WARNING
    reraise=True,
)

# --- Helper to get duration using mutagen or tinytag ---
def get_audio_duration(file_path):
    """Attempts to get audio/video duration using mutagen, then tinytag."""
    duration = None

    if MutagenFile is not None:
        try:
            audio = MutagenFile(file_path)
            if audio and hasattr(audio.info, 'length') and audio.info.length is not None:
                duration = audio.info.length
                logger.debug(f"Mutagen detected duration for {os.path.basename(file_path)}: {duration:.2f} seconds")
                return duration
            logger.debug(f"Mutagen could not detect duration for {os.path.basename(file_path)}")
        except Exception as e:
            logger.debug(f"Error detecting duration for {os.path.basename(file_path)} using mutagen: {e}")

    if TinyTag is not None:
        try:
            tag = TinyTag.get(file_path)
            if tag and tag.duration is not None:
                duration = tag.duration
                logger.debug(f"Tinytag detected duration for {os.path.basename(file_path)}: {duration:.2f} seconds")
                return duration
            logger.debug(f"Tinytag could not detect duration for {os.path.basename(file_path)}")
        except Exception as e:
            logger.debug(f"Error detecting duration for {os.path.basename(file_path)} using tinytag: {e}")

    logger.debug(f"Failed to detect duration for {os.path.basename(file_path)} using both mutagen and tinytag.")
    return None


def format_duration(seconds):
    """Formats duration in seconds to HH:MM:SS."""
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_output_data(filepath):
    """Loads data from a JSON file, handling errors."""
    data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, dict):
                    data = loaded_data
                else:
                     logger.warning(f"Existing file '{filepath}' does not contain a JSON object. Starting with an empty structure.")
        except json.JSONDecodeError:
            logger.warning(f"Could not decode existing JSON file '{filepath}'. Starting with an empty structure.")
        except Exception as e:
            logger.warning(f"Error reading existing JSON file '{filepath}': {e}. Starting with an empty structure.")
    return data


def save_output_data(filepath, data):
    """Saves data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Data saved to '{filepath}'.") # Debug logs are still logged
    except Exception as e:
        logger.error(f"Error saving data to '{filepath}': {e}")


def get_assemblyai_api_key():
    """Gets the AssemblyAI API key from environment variables or asks the user with masking. Caches the key globally."""
    global _ASSEMBLYAI_API_KEY
    if _ASSEMBLYAI_API_KEY:
        return _ASSEMBLYAI_API_KEY

    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        # Use standard print for getpass prompt
        print("Please enter your AssemblyAI API key for transcription: ", end="", flush=True)
        api_key = getpass.getpass("")
    
    _ASSEMBLYAI_API_KEY = api_key
    return api_key


def get_xai_api_key():
    """Gets the xAI API key from environment variables or asks the user with masking. Caches the key globally."""
    global _XAI_API_KEY
    if _XAI_API_KEY:
        return _XAI_API_KEY

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        print("Please enter your xAI API key for summarization: ", end="", flush=True)
        api_key = getpass.getpass("")
    
    _XAI_API_KEY = api_key
    return api_key

# --- Function to Download File from URL ---
def download_file_from_url(url, download_folder=DOWNLOAD_FOLDER, overall_pbar=None):
    """Downloads a file from a given URL to a specified folder with improved robustness."""
    os.makedirs(download_folder, exist_ok=True)
    
    local_filename = None
    file_extension = None

    try:
        # Perform a HEAD request first to get headers like Content-Disposition and Content-Type
        # Allow redirects as the final URL might have the correct headers or extension
        response_head = requests.head(url, allow_redirects=True, timeout=10) # Added timeout
        response_head.raise_for_status() # Check for HTTP errors on HEAD request

        # 1. Try Content-Disposition
        if 'content-disposition' in response_head.headers:
            disposition = response_head.headers['content-disposition']
            filenames = re.findall('filename="?([^"]+)"?', disposition)
            if filenames:
                local_filename = filenames[0]
                logger.debug(f"Filename from Content-Disposition: {local_filename}")

        # 2. If no filename from Content-Disposition, parse from URL
        if not local_filename:
            local_filename = url.split('/')[-1].split('?')[0] # Remove query params for filename
            logger.debug(f"Filename from URL path: {local_filename}")

        # Sanitize the filename
        local_filename = re.sub(r'[^\w.-]', '_', local_filename)
        
        # Extract extension
        if local_filename and '.' in local_filename:
            name_part, ext_part = os.path.splitext(local_filename)
            if ext_part: # Ensure ext_part is not just '.'
                 file_extension = ext_part
                 local_filename = name_part # Keep local_filename as base name for now
        
        # If no extension from filename, try from URL
        if not file_extension:
            url_path = requests.utils.urlparse(url).path
            name_part_url, ext_part_url = os.path.splitext(url_path)
            if ext_part_url:
                file_extension = ext_part_url
                logger.debug(f"Extension from URL: {file_extension}")

        # If still no extension, try Content-Type (MIME type to extension mapping)
        if not file_extension and 'content-type' in response_head.headers:
            content_type = response_head.headers['content-type'].split(';')[0]
            # Basic mapping, can be expanded
            mime_map = {
                'audio/mpeg': '.mp3', 'audio/wav': '.wav', 'audio/aac': '.aac', 'audio/ogg': '.ogg',
                'audio/flac': '.flac', 'audio/x-m4a': '.m4a', 'video/mp4': '.mp4', 'video/quicktime': '.mov',
                'video/x-ms-wmv': '.wmv', 'video/x-matroska': '.mkv', 'video/webm': '.webm'
            }
            if content_type in mime_map:
                file_extension = mime_map[content_type]
                logger.debug(f"Extension from Content-Type ({content_type}): {file_extension}")

        # 3. Fallback for filename body if still problematic
        if not local_filename or local_filename.startswith('.') or local_filename in ('.', '..'):
            local_filename = f"downloaded_file_{int(time.time())}"
            logger.debug(f"Using fallback filename body: {local_filename}")

        # Append extension; if no extension found, default to .tmp or leave extensionless
        # For this script, a media extension is important, so if none, it might be an issue.
        # Defaulting to .tmp if totally unknown, or consider not adding if truly not found.
        file_extension = file_extension if file_extension and file_extension.startswith('.') else ".download" # Ensure extension starts with a dot
        
        final_filename = local_filename + file_extension
        filepath = os.path.join(download_folder, final_filename)

        # Use print or tqdm.write for INFO messages in non-verbose
        if logger.isEnabledFor(logging.INFO): # Check if in non-verbose mode
            if overall_pbar and not overall_pbar.disable:
                overall_pbar.write(f"Downloading {os.path.basename(filepath)}...")
                overall_pbar.refresh()
            else:
                print(f"Downloading {os.path.basename(filepath)}...", flush=True)
        logger.debug(f"Attempting to download from {url} to {filepath}")

        # Use tqdm for download progress - disabled if not in INFO mode
        with tqdm(total=None, unit='B', unit_scale=True, desc=f"Download Progress", leave=False, disable=not logger.isEnabledFor(logging.INFO), file=sys.stdout) as pbar:
            # Get total size from HEAD request if available
            total_size = int(response_head.headers.get('content-length', 0))
            if total_size > 0:
                pbar.total = total_size
            else:
                pbar.total = None # Will show progress without percentage

            with requests.get(url, stream=True, timeout=60) as r: # Added timeout
                r.raise_for_status() # Check for HTTP errors on GET request

                block_size = 8192
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:
                            pbar.update(len(chunk))
        
        # Clear the download progress bar line if it was active and not completed
        if logger.isEnabledFor(logging.INFO) and (pbar.total is None or pbar.n < (pbar.total or 1)):
             print(" " * 80, end='\r', flush=True)

        if logger.isEnabledFor(logging.INFO):
            if overall_pbar and not overall_pbar.disable:
                overall_pbar.write(f"Download complete: {os.path.basename(filepath)}")
                overall_pbar.refresh()
            else:
                print(f"Download complete: {os.path.basename(filepath)}", flush=True)
        logger.debug(f"Downloaded file path: {filepath}")
        return filepath

    except requests.exceptions.Timeout as e:
        logger.error(f"Download timed out for URL: {url}. Error: {e}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error while downloading URL: {url}. Error: {e}")
        return None
    except requests.exceptions.HTTPError as e: # Catch HTTP errors from HEAD or GET
        logger.error(f"HTTP error downloading file from {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
             logger.debug(f"Download HTTP Status: {e.response.status_code}")
             try:
                 logger.debug(f"Download Response Body: {e.response.json()}")
             except json.JSONDecodeError:
                 logger.debug(f"Download Response Body (non-JSON): {e.response.text if e.response.text else 'Empty'}")
        return None
    except requests.exceptions.RequestException as e: # Catch other request-related errors
        logger.error(f"Error downloading file from {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
             logger.debug(f"Download HTTP Status: {e.response.status_code}")
             try:
                 logger.debug(f"Download Response Body: {e.response.json()}")
             except json.JSONDecodeError:
                 # Check if e.cause is available and has response for more specific logging if needed
                 response_text = e.response.text if hasattr(e.response, 'text') else 'Empty'
                 if not response_text and hasattr(e, 'cause') and hasattr(e.cause, 'response') and hasattr(e.cause.response, 'text'):
                     response_text = e.cause.response.text
                 logger.debug(f"Download Response Body (non-JSON): {response_text if response_text else 'Empty'}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during download of {url}: {e}")
        return None


# --- AssemblyAI Transcription Function with Retry Workaround ---
def transcribe_audio_with_diarization(audio_file_path, assemblyai_api_key, num_speakers=None, force=False, overall_pbar=None, file_status_desc=""):
    """
    Transcribes a local audio/video file with speaker diarization using AssemblyAI.
    Returns transcript segments and detected duration.
    overall_pbar and file_status_desc are used to update the tqdm description.
    """
    base_filename = os.path.basename(audio_file_path)
    output_dir = os.path.dirname(audio_file_path) or '.'
    output_filename = os.path.join(output_dir, os.path.splitext(base_filename)[0] + TRANSCRIPT_FILENAME_SUFFIX)

    output_data = load_output_data(output_filename)

    if "transcript" in output_data and not force:
        logger.warning(
            f"Transcript data found in '{output_filename}'. Skipping transcription (use --force-transcribe to re-run)."
        ) # Keep warning
        existing_segments = output_data.get("transcript", {}).get("segments")
        existing_duration = output_data.get("audio_info", {}).get("duration_seconds")
        return existing_segments, existing_duration

    if not assemblyai_api_key:
        logger.error("Error: AssemblyAI API key not provided. Exiting transcription.") # Keep error
        return None, None

    assemblyai.settings.api_key = assemblyai_api_key

    logger.debug(f"Step: Starting AssemblyAI transcription...") # Debug only
    logger.debug(f"Sending '{audio_file_path}' for transcription with {num_speakers} expected speakers.") # Debug only
    try:
        @assemblyai_retry_settings
        def _transcribe_core(file_path, num_speakers_expected, pbar, file_desc):
            config = assemblyai.TranscriptionConfig(
                speaker_labels=True,
                speakers_expected=num_speakers_expected,
            )
            transcriber = assemblyai.Transcriber(config=config)

            logger.debug("Submitting transcription job to AssemblyAI...") # Debug only

            transcript = transcriber.transcribe(file_path)
            logger.debug(f"Transcription job submitted. Job ID: {transcript.id}") # Debug only

            # Update tqdm description or print status dynamically
            while transcript.status in ['queued', 'processing']:
                status_msg = f"[Transcription: {transcript.status.value}]"
                if logger.isEnabledFor(logging.INFO): # Only update dynamic status in non-verbose
                    if pbar and not pbar.disable:
                         pbar.set_description(f"{file_desc} {status_msg}")
                         pbar.refresh()
                    else:
                         print(f"{file_desc} {status_msg}", end='\r', flush=True)
                logger.debug(f"Polling status for job {transcript.id}: {transcript.status.value}") # Debug polling status

                time.sleep(5)
                transcript = transcriber.get_transcript(transcript.id)


            # Ensure final status is reflected in description before next step
            final_status_msg = f"[Transcription: {transcript.status.value}]"
            if logger.isEnabledFor(logging.INFO): # Only update dynamic status in non-verbose
                 if pbar and not pbar.disable:
                      pbar.set_description(f"{file_desc} {final_status_msg}")
                      pbar.refresh()
                 else:
                      print(f"{file_desc} {final_status_msg}", end='\r', flush=True)
                      # Clear the line only in fallback mode if it was printed
                      # Only clear if transcription wasn't completed successfully and in non-verbose
                      if transcript.status != 'completed' and logger.isEnabledFor(logging.INFO):
                           print(" " * (len(file_desc) + len(final_status_msg) + 5), end='\r', flush=True)


            if transcript.status == 'completed':
                 logger.debug("Transcription status: Completed.") # Debug only
                 detected_duration = None
                 if hasattr(transcript, 'duration') and transcript.duration is not None:
                     detected_duration = transcript.duration
                     # Duration is informational, handled in process_single_file logging summary
                 # Note: duration is NOT saved to output_data here, but in process_single_file using AA or local duration

                 return transcript, detected_duration
            else:
                 logger.error(f"Transcription failed with status: {transcript.status.value}") # Keep error
                 if transcript.error:
                     logger.error(f"Error details: {transcript.error}") # Keep error
                 return None, None

        # Pass audio_file_path to _transcribe_core
        transcript_object, aa_detected_duration = _transcribe_core(audio_file_path, num_speakers, overall_pbar, file_status_desc)

        if transcript_object and transcript_object.utterances:
            segments = []
            for utterance in transcript_object.utterances:
                segments.append({"speaker": f"Speaker {utterance.speaker}", "text": utterance.text})

            if "transcript" not in output_data:
                output_data["transcript"] = {}
            output_data["transcript"]["segments"] = segments
            detected_speaker_count = len(set(s['speaker'] for s in segments))
            output_data["transcript"]["detected_speaker_count"] = detected_speaker_count
            output_data["transcript"]["expected_speakers_input"] = num_speakers
            # Speaker count is informational, handled in process_single_file logging summary
            save_output_data(output_filename, output_data)
            logger.debug(f"Step: Transcription data processed and saved.") # Debug only
            return segments, aa_detected_duration # Return AA duration here
        elif transcript_object is not None:
            logger.warning("Transcription completed, but no utterances found (possibly empty audio).") # Keep warning
            if "transcript" not in output_data:
                 output_data["transcript"] = {}
            output_data["transcript"]["segments"] = []
            output_data["transcript"]["detected_speaker_count"] = 0
            output_data["transcript"]["expected_speakers_input"] = num_speakers
            save_output_data(output_filename, output_data)
            return [], aa_detected_duration # Return AA duration here

        else:
            logger.error("Transcription failed.") # Keep error
            return None, None

    except tenacity.RetryError as e:
        logger.error(f"Failed to transcribe audio after multiple retries: {e}") # Keep error
        if e.cause and isinstance(e.cause, requests.exceptions.RequestException):
             if hasattr(e.cause, "response") and hasattr(e.cause.status_code):
                 logger.error(f"Last attempt failed with HTTP status: {e.cause.response.status_code}") # Keep error
                 try:
                     error_details = e.cause.response.json()
                     logger.error(f"Error details: {error_details}") # Keep error
                 except json.JSONDecodeError:
                     logger.error(f"Error response body: {e.cause.response.text}") # Keep error
             else:
                 logger.error(f"Last attempt failed with request exception: {e.cause}") # Keep error
        return None, None
    except requests.exceptions.RequestException as e:
        logger.error(f"A requests error occurred during AssemblyAI transcription that was not retried: {e}") # Keep error
        if isinstance(e, requests.exceptions.HTTPError):
            logger.error(f"HTTP Status: {e.response.status_code}") # Keep error
            try:
                logger.error(f"Response Body: {e.response.json()}") # Keep error
            except json.JSONDecodeError:
                logger.error(f"Response Body (non-JSON): {e.cause.response.text}") # Keep error
        return None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during transcription: {e}") # Keep error
        return None, None


# --- Retryable xAI Models List Call ---
@common_retry_settings
@tenacity.retry(
    retry=retry_if_transient_http_error | tenacity.retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_xai_models_with_retry():
    """Fetches xAI models with retry logic."""
    api_key = get_xai_api_key()
    if not api_key:
        logger.error("xAI API key not provided for fetching models.") # Keep error
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    logger.debug(f"Fetching xAI models from {XAI_MODELS_ENDPOINT}") # Debug only
    response = requests.get(XAI_MODELS_ENDPOINT, headers=headers)
    logger.debug(f"xAI Models API response status: {response.status_code}") # Debug only
    response.raise_for_status()
    return response.json()


def get_latest_xai_model():
    """
    Fetches the list of available xAI models, logs the list (at debug level),
    and selects a model based on priority. Gets API key internally and caches the selected model.
    """
    global _SELECTED_XAI_MODEL
    if _SELECTED_XAI_MODEL:
        return _SELECTED_XAI_MODEL

    api_key = get_xai_api_key()
    if not api_key:
         logger.warning("xAI API key not available for model selection. Skipping.") # Keep warning
         return None

    logger.debug("Step: Fetching list of available xAI models...") # Debug only
    available_models = []
    try:
        models_data = fetch_xai_models_with_retry()

        if not models_data or "data" not in models_data or not isinstance(models_data["data"], list):
            logger.error("Error: Unexpected response format from xAI models endpoint.") # Keep error
            return None

        available_models = [model["id"] for model in models_data["data"] if "id" in model]

    except tenacity.RetryError as e:
        logger.error(f"Failed to fetch xAI models after multiple retries: {e}") # Keep error
        return None
    except Exception as e:
        logger.error(f"An error occurred while fetching or processing xAI models: {e}") # Keep error
        return None

    if available_models:
        logger.debug("\nAvailable models from xAI:") # Debug only
        for model_id in available_models:
            logger.debug(f"- {model_id}") # Debug only
        logger.debug("-" * 20) # Debug only
    else:
        logger.warning("\nNo models retrieved from xAI.") # Keep warning
        return None

    grok_pattern = re.compile(r"^grok-(\d+)(?:-.*)?$")
    grok_models_with_numbers = []

    for model_id in available_models:
        match = grok_pattern.match(model_id)
        if match:
            number = int(match.group(1))
            grok_models_with_numbers.append((number, model_id))

    selected_model = None

    if grok_models_with_numbers:
        highest_number = max([num for num, _ in grok_models_with_numbers])

        prioritized_patterns = [
            f"grok-{highest_number}-mini-fast-beta",
            f"grok-{highest_number}-beta",
            f"grok-{highest_number}",
        ]

        logger.debug(f"Highest grok version found: {highest_number}. Checking prioritized patterns:") # Debug only
        for pattern in prioritized_patterns:
            if pattern in available_models:
                selected_model = pattern
                logger.debug(f"Step: Selected model matching pattern: {selected_model}") # Debug only
                _SELECTED_XAI_MODEL = selected_model
                return selected_model

        logger.debug(f"None of the prioritized grok-{highest_number} patterns found.") # Debug only

    else:
        logger.debug("No models matching the 'grok-N' pattern found.") # Debug only

    fallback_specific = "grok-3-latest"
    if fallback_specific in available_models:
        selected_model = fallback_specific
        logger.debug(f"Step: Falling back to specific model: {selected_model}") # Debug only
        _SELECTED_XAI_MODEL = selected_model
        return fallback_specific

    if available_models:
        selected_model = available_models[0]
        logger.warning( # Keep warning for fallback model selection
            f"Neither prioritized grok patterns nor '{fallback_specific}' found. Falling back to selecting the first available model: {selected_model}"
        )
        _SELECTED_XAI_MODEL = selected_model
        return selected_model
    else:
        logger.error("No available models found to select from.") # Keep error
        return None


def extract_speaker_name_mapping(summary_text):
    """
    Analyzes the summary text to find potential speaker name mappings.
    """
    if not summary_text:
        return {}

    name_mapping = {}

    pattern1 = re.compile(r"(\b[A-Z][a-z]+(?:['\s-][A-Z][a-z]+)*|\b[A-Z]+\b)\s+\(Speaker ([A-Z]+)\)")
    matches1 = pattern1.findall(summary_text)
    for name, speaker_label_suffix in matches1:
        original_label = f"Speaker {speaker_label_suffix}"
        name_mapping[original_label] = name.strip()

    pattern2 = re.compile(r"Speaker ([A-Z]+)\s+\((\b[A-Z][a-z]+(?:['\s-][A-Z][a-z]+)*|\b[A-Z]+\b)\)")
    matches2 = pattern2.findall(summary_text)
    for speaker_label_suffix, name in matches2:
        original_label = f"Speaker {speaker_label_suffix}"
        name_mapping[original_label] = name.strip()

    pattern3 = re.compile(r"(\b[A-Z][a-z]+(?:['\s-][A-Z][a-z]+)*|\b[A-Z]+\b)\s+\(Speaker ([A-Z]+)\)\s+said")
    matches3 = pattern3.findall(summary_text)
    for name, speaker_label_suffix in matches3:
         original_label = f"Speaker {speaker_label_suffix}"
         name_mapping[original_label] = name.strip()

    logger.debug(f"Extracted potential speaker name mapping: {name_mapping}") # Debug only
    return name_mapping


# --- Retryable xAI Messages Post Call ---
@common_retry_settings
@tenacity.retry(
    retry=retry_if_transient_http_error | tenacity.retry_if_exception_type(requests.exceptions.RequestException)
)
def post_xai_messages_with_retry(api_key, model_id, prompt_text, max_tokens):
    """Posts message to xAI API with retry logic."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
    }
    logger.debug(f"Sending summarization request to {XAI_MESSAGES_ENDPOINT} with payload: {payload}") # Debug only
    response = requests.post(XAI_MESSAGES_ENDPOINT, headers=headers, json=payload)
    logger.debug(f"xAI Messages API response status: {response.status_code}") # Debug only
    response.raise_for_status()
    return response.json()


def summarize_transcript_with_xai(
    transcript_segments,
    audio_file_path,
    xai_api_key,
    model_id,
    audio_datetime_str=None,
    force=False,
    max_tokens=10000,
    overall_pbar=None, # Accept overall_pbar
    file_status_desc="" # Accept file status description prefix
):
    """
    Summarizes the given transcript segments using a specified xAI LLM model.
    overall_pbar and file_status_desc are used to update the tqdm description.
    """
    base_filename = os.path.basename(audio_file_path)
    output_dir = os.path.dirname(audio_file_path) or '.'
    output_filename = os.path.join(output_dir, os.path.splitext(base_filename)[0] + TRANSCRIPT_FILENAME_SUFFIX)

    output_data = load_output_data(output_filename)

    if ("summary" in output_data or "xai_usage" in output_data) and not force:
        logger.warning(
            f"Summary or xAI usage data found in '{output_filename}'. Skipping summarization (use --force-summarize to re-run)."
        ) # Keep warning
        return output_data.get("summary"), output_data.get("xai_usage") # Return existing usage if skipping

    if not xai_api_key:
        logger.error("Error: xAI API key not provided. Exiting summarization.") # Keep error
        return None, None

    if not XAI_MESSAGES_ENDPOINT:
        logger.error("Error: xAI Messages API endpoint not configured. Exiting summarization.") # Keep error
        return None, None

    if not transcript_segments:
         logger.warning("No transcript segments available for summarization. Skipping.") # Keep warning
         return None, None

    full_transcript_text = "\n".join([f"{item['speaker']}: {item['text']}" for item in transcript_segments])

    prompt_text = f"""Summarize the following conversation transcript. Format the output in Markdown.

Identify speakers by their names from the transcript content whenever possible. If a speaker's name cannot be determined from the transcript content, use their assigned label (e.g., Speaker A, Speaker B) from the transcript.

When you first mention a speaker in the summary (either by name or label), format it as follows:
- If the name is known: "Name (Speaker X)" (e.g., "Bill (Speaker A)")
- If the name is unknown: "Speaker X (name not specified)" (e.g., "Speaker B (name not specified)")
For all *subsequent* mentions of that same speaker in the summary, use only their identified name or their label (e.g., just "Bill" or just "Speaker A"), without any additional comments in brackets.

Focus on key discussion points, using a bulleted list (`- `). Attribute points to speakers using their name or label as described above (applying the first mention vs. subsequent mention rule).

If the conversation resembles a meeting, include relevant sections formatted as Markdown, such as:
- **Date:** {audio_datetime_str if audio_datetime_str else 'N/A'}
- **Attendees:** [List identified speakers by name, or label if name is unknown. Use the first mention format defined above for each attendee listed here. List them in order of their first appearance in the transcript.]
- **Topics Discussed:** (Reference the bulleted list of key points)
- **Decisions:** [List any decisions identified]
- **Action Items:** [List any action items identified]

If it is not a meeting, provide a concise summary using a Markdown bulleted list for key topics.

Transcript:
{full_transcript_text}
"""
    logger.debug(f"Prompt sent to xAI:\n---\n{prompt_text}\n---") # Debug only

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
    }

    # Update tqdm description or print status dynamically
    status_msg = "[Summarization: Sending to xAI]"
    if logger.isEnabledFor(logging.INFO): # Only update dynamic status in non-verbose
         if overall_pbar and not overall_pbar.disable:
              overall_pbar.set_description(f"{file_status_desc} {status_msg}")
              overall_pbar.refresh()
         else:
              print(f"{file_status_desc} {status_msg}", end='\r', flush=True)


    logger.debug(f"Step: Sending summarization request to xAI...") # Debug only
    logger.debug(f"Using model: {model_id}, max_tokens={max_tokens}") # Debug only
    try:
        summary_data = post_xai_messages_with_retry(xai_api_key, model_id, prompt_text, max_tokens)
        logger.debug(f"xAI API response data: {summary_data}") # Debug only

        summary = None
        usage = None

        if summary_data and "content" in summary_data and isinstance(summary_data["content"], list):
            for block in summary_data["content"]:
                if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                    summary = block["text"]
                    break

        if summary_data and "usage" in summary_data and isinstance(summary_data["usage"], dict):
            usage = summary_data["usage"]
            logger.debug(f"Raw xAI usage data: {usage}") # Debug only


        if summary:
            output_data["summary"] = summary
            logger.debug(f"Step: Summary generated successfully.") # Debug only
            logger.debug(f"Generated Summary:\n---\n{summary}\n---") # Debug only


            if usage is not None:
                output_data["xai_usage"] = usage
                logger.debug(f"xAI usage data recorded: {usage}") # Debug only
                # XAI Tokens Used is informational - handled in process_single_file logging summary


            logger.debug("Step: Attempting to extract speaker name mapping from summary...") # Debug only
            speaker_name_mapping = extract_speaker_name_mapping(summary)

            if speaker_name_mapping:
                # Identified speaker names is informational - handled in process_single_file logging summary
                if (
                    "transcript" in output_data
                    and isinstance(output_data["transcript"], dict)
                    and "segments" in output_data["transcript"]
                    and isinstance(output_data["transcript"]["segments"], list)
                ):
                    logger.debug("Updating transcript segments with identified names...") # Debug only
                    for segment in output_data["transcript"]["segments"]:
                        original_speaker_label = segment.get("speaker")
                        if original_speaker_label and original_speaker_label in speaker_name_mapping:
                             segment["speaker"] = speaker_name_mapping[original_speaker_label]
                    logger.debug("Transcript segments updated.") # Debug only
                else:
                    logger.warning(
                        "Transcript segments not found in output_data in expected format for name mapping update."
                    ) # Keep warning
            else:
                logger.debug("No speaker names identified from the summary for transcript update.") # Debug only

            save_output_data(output_filename, output_data)
            logger.debug(f"Step: Summary, usage, and updated transcript data processed and saved.") # Debug only

            return summary, usage # Return both summary and usage

        else:
            logger.error(f"Error: Could not extract summary text from xAI API response. Response data: {summary_data}") # Keep error
            if usage is not None:
                 output_data["xai_usage"] = usage
                 logger.debug(f"xAI usage data recorded despite summary extraction failure: {usage}") # Debug only
                 save_output_data(output_filename, output_data)
                 logger.warning(f"Usage data saved to '{output_filename}'.") # Keep warning for saving usage data

            return None, usage # Return usage even if summary extraction fails

    except tenacity.RetryError as e:
        logger.error(f"Failed to generate summary after multiple retries: {e}") # Keep error
        return None, None
    except Exception as e:
        logger.error(f"An error occurred during summarization: {e}") # Keep error
        return None, None

# --- Function to process a single file ---
def process_single_file(file_path, assemblyai_api_key, xai_api_key, selected_xai_model, expected_speakers, force_transcribe, force_summarize, max_tokens, overall_pbar=None, file_index=None, total_files=None):
    """Processes a single audio/video file."""

    # Prepare file status description prefix for tqdm bar
    file_status_prefix = f"File {file_index}/{total_files}: {os.path.basename(file_path)}" if overall_pbar else f"File: {os.path.basename(file_path)}"

    # Log file processing start in verbose. In non-verbose, it's implicit in tqdm description.
    if logger.isEnabledFor(logging.DEBUG): # Only log in verbose mode
         logger.info(f"--- Starting processing for {os.path.basename(file_path)} ---")


    audio_datetime_str = None
    try:
        mtime = os.path.getmtime(file_path)
        audio_datetime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        # Date/Time is informational - handled in final logging summary if needed


    except OSError as e:
        logger.warning(f"Could not get file modification time for '{file_path}': {e}") # Keep warning
    except Exception as e:
        logger.warning(f"An unexpected error occurred while getting audio date/time: {e}") # Keep warning

    # Corrected variable name here from audio_file_path to file_path
    base_filename = os.path.basename(file_path)
    output_dir = os.path.dirname(file_path) or '.'
    output_filename = os.path.join(output_dir, os.path.splitext(base_filename)[0] + TRANSCRIPT_FILENAME_SUFFIX)
    output_data = load_output_data(output_filename)

    if "audio_info" not in output_data:
         output_data["audio_info"] = {}
    output_data["audio_info"]["original_path"] = file_path
    if audio_datetime_str:
         output_data["audio_info"]["recorded_datetime"] = audio_datetime_str
    save_output_data(output_filename, output_data)


    # --- Transcription ---
    logger.debug(f"Step: Starting transcription process...") # Debug only
    # Update tqdm description or print status for transcription
    status_msg = "[Transcription]"
    if logger.isEnabledFor(logging.INFO): # Only update dynamic status in non-verbose
         if overall_pbar and not overall_pbar.disable:
              overall_pbar.set_description(f"{file_status_prefix} {status_msg}")
              overall_pbar.refresh()
         else:
              print(f"{file_status_prefix} {status_msg}", end='\r', flush=True)


    transcript_segments, aa_detected_duration = transcribe_audio_with_diarization(
        file_path, assemblyai_api_key, expected_speakers, force=force_transcribe, overall_pbar=overall_pbar, file_status_desc=file_status_prefix
    )

    # --- Determine duration for overall progress bar update ---
    duration_for_pbar_update = None
    update_source = "None"

    if aa_detected_duration is not None and aa_detected_duration > 0:
        duration_for_pbar_update = aa_detected_duration
        update_source = "AssemblyAI"
    else:
        # If AA duration is not available or zero, try local duration for pbar update
        local_duration = get_audio_duration(file_path) # Get local duration for this file
        if local_duration is not None and local_duration > 0:
            duration_for_pbar_update = local_duration
            update_source = "Local"
            # If using local duration for pbar, also save it to output data if AA didn't provide one
            if "duration_seconds" not in output_data.get("audio_info", {}):
                 output_data.setdefault("audio_info", {})["duration_seconds"] = local_duration
                 save_output_data(output_filename, output_data)


    # --- Update overall progress bar ---
    if overall_pbar: # Check if pbar exists
        if duration_for_pbar_update is not None and duration_for_pbar_update > 0: # Also check if duration is positive
             logger.debug(f"Updating overall_pbar with duration from {update_source}: {duration_for_pbar_update:.2f}") # Debug print
             overall_pbar.update(duration_for_pbar_update)
             overall_pbar.refresh()
        else:
             logger.debug(f"Valid duration ({duration_for_pbar_update}) from {update_source} not available, cannot update overall_pbar for {os.path.basename(file_path)}.") # Debug print
             # Inform the user in non-verbose mode if duration update is skipped
             if logger.isEnabledFor(logging.INFO):
                  if overall_pbar and not overall_pbar.disable:
                       overall_pbar.write(f"Warning: Could not get valid duration ({update_source}) for {os.path.basename(file_path)}. Overall progress bar may not update accurately for this file.")
                       overall_pbar.refresh()
                  # No fallback print needed here, as the lack of update is the message


    elif logger.isEnabledFor(logging.DEBUG): # If no overall pbar but in verbose, still log detected duration
        logger.debug(f"Detected duration for single file from {update_source}: {duration_for_pbar_update:.2f}")


    # --- Summarization ---
    xai_usage_data = None # Variable to hold xAI usage data for final summary log
    summary = None # Variable to hold summary

    if transcript_segments is not None:
        logger.debug(f"Step: Starting summarization process...") # Debug only
        # Update tqdm description or print status for summarization
        status_msg = "[Summarization]"
        if logger.isEnabledFor(logging.INFO): # Only update dynamic status in non-verbose
             if overall_pbar and not overall_pbar.disable:
                  overall_pbar.set_description(f"{file_status_prefix} {status_msg}")
                  overall_pbar.refresh()
             else:
                  print(f"{file_status_prefix} {status_msg}", end='\r', flush=True)


        if xai_api_key: # Check if xAI key is available
            if selected_xai_model: # Check if a model was successfully selected and passed
                summary, xai_usage_data = summarize_transcript_with_xai( # Capture usage data
                    transcript_segments,
                    file_path,
                    xai_api_key, # Pass the key
                    selected_xai_model, # Pass the pre-fetched model
                    audio_datetime_str=audio_datetime_str,
                    force=force_summarize,
                    max_tokens=max_tokens,
                    overall_pbar=overall_pbar, # Pass overall_pbar
                    file_status_desc=file_status_prefix # Pass file status prefix
                )
            else:
                logger.warning("A suitable xAI model was not provided or could not be determined. Skipping summarization.") # Keep warning
        else:
            logger.warning("xAI API key not available. Skipping summarization.") # Keep warning
    else:
        logger.warning("Transcription failed or skipped. Skipping summarization.") # Keep warning


    # Clear the dynamic line completely using the prefix length
    # Only clear if an overall bar was NOT active, as tqdm manages its own line
    if logger.isEnabledFor(logging.INFO) and (overall_pbar is None or overall_pbar.disable):
         # Estimate line length to clear
         est_len = len(file_status_prefix) + 40 # Add some buffer
         print(" " * est_len, end='\r', flush=True)


    # --- Log final summary details for the file ---
    # This block handles summary details for both verbose and non-verbose
    summary_lines = []
    summary_lines.append(f"--- Finished processing {os.path.basename(file_path)} ---")

    # Reload output data to get latest duration and speakers and usage if needed
    output_data_final = load_output_data(output_filename)
    # Get duration from output data, which now prefers local if AA was None/0
    final_duration_seconds = output_data_final.get("audio_info", {}).get("duration_seconds")
    final_speakers = output_data_final.get("transcript", {}).get("detected_speaker_count")
    final_xai_usage = output_data_final.get("xai_usage", {})


    if audio_datetime_str:
         summary_lines.append(f"Date/Time: {audio_datetime_str}")

    # Use the duration from output data for the final summary display
    if final_duration_seconds is not None:
         summary_lines.append(f"Duration: {format_duration(final_duration_seconds)}")


    if final_speakers is not None:
         summary_lines.append(f"Speakers detected: {final_speakers}")

    if final_xai_usage:
        prompt_t = final_xai_usage.get('input_tokens', 'N/A')
        completion_t = final_xai_usage.get('output_tokens', 'N/A')
        total_t = 'N/A'
        if isinstance(prompt_t, int) and isinstance(completion_t, int):
            total_t = prompt_t + completion_t
        summary_lines.append(f"xAI Tokens: In:{prompt_t}, Out:{completion_t}, Total:{total_t}")


    summary_lines.append(f"Output saved to: {output_filename}")

    # Print the summary block using appropriate method
    if logger.isEnabledFor(logging.INFO): # In non-verbose, print above the next tqdm bar
         if overall_pbar and not overall_pbar.disable:
              for line in summary_lines:
                   overall_pbar.write(line)
              overall_pbar.refresh()
         else: # In non-verbose single file, just print
              for line in summary_lines:
                   print(line, flush=True)

    else: # In verbose mode, use logger.info for the summary block
         for line in summary_lines:
              logger.info(line)


    return duration_for_pbar_update # Return the duration actually used for the pbar update


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Transcribe and summarize audio/video from file, URL, or folder using AssemblyAI and xAI.")
    parser.add_argument(
        "input_source",
        help="Path to a local audio/video file, a URL to a remote file, or a path to a folder containing audio/video files."
    )
    parser.add_argument(
        "-s",
        "--speakers",
        type=int,
        default=3,
        help="Expected number of speakers (optional, default is 3). Applies to all files if input is a folder.",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Force re-transcription even if transcript data exists.",
    )
    parser.add_argument(
        "--force-summarize",
        action="store_true",
        help="Force re-summarization even if summary exists.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum number of tokens for the xAI summarization model (default: 10000).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for detailed output.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Number of concurrent workers for processing files in folder mode (default: 3).",
    )
    args = parser.parse_args()

    input_source = args.input_source
    expected_speakers = args.speakers
    force_transcribe = args.force_transcribe
    force_summarize = args.force_summarize
    max_tokens = args.max_tokens
    verbose_logging = args.verbose
    num_workers = args.num_workers

    # --- Configure Logging Handler and Formatter based on verbosity ---
    # Remove existing handlers first
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Add a standard console handler for verbose mode or the custom handler for non-verbose WARNING/ERROR
    if verbose_logging:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("requests").setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter_verbose)
        logger.addHandler(console_handler)
        # Ensure requests/urllib3 also use this handler
        for req_handler in logging.getLogger("requests").handlers: logging.getLogger("requests").removeHandler(req_handler)
        for url_handler in logging.getLogger("urllib3").handlers: logging.getLogger("urllib3").removeHandler(url_handler)
        logging.getLogger("requests").addHandler(console_handler)
        logging.getLogger("urllib3").addHandler(console_handler)


        tenacity.before_sleep_log(logger, logging.DEBUG) # Log retries at DEBUG in verbose

    else:
        # Non-verbose: Logger only for WARNING/ERROR, use TqdmWarningErrorHandler
        # Set logger level to INFO so logger.isEnabledFor(logging.INFO) is true for conditional prints
        # However, our TqdmWarningErrorHandler still only processes WARNING/ERROR
        logger.setLevel(logging.INFO) # Set to INFO so isEnabledFor works

        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        # The warning_error_handler instance needs to be accessible globally or passed
        # Let's create it here and pass it to the main processing logic if needed
        # However, directly updating its pbar attribute in the main block is simpler
        # as done below. The handler is already added to the logger here.
        # warning_error_handler = TqdmWarningErrorHandler(level=logging.WARNING) # Instance created globally now
        warning_error_handler.setLevel(logging.WARNING) # Ensure correct level
        warning_error_handler.setFormatter(formatter_non_verbose) # Ensure correct formatter
        # Handler is already added above...


        tenacity.before_sleep_log(logger, logging.WARNING) # Log retries at WARNING in non-verbose


    # Get API keys AFTER configuring handlers so prompts use correct output
    # These will now use the caching mechanism
    assemblyai_api_key = get_assemblyai_api_key()
    xai_api_key = get_xai_api_key() 
    selected_xai_model = None # Initialize

    if not assemblyai_api_key:
        logger.error("AssemblyAI API key is required for transcription. Please set ASSEMBLYAI_API_KEY environment variable or provide it when prompted.")
        sys.exit(1)

    # Initial processing message - use print/tqdm.write in non-verbose
    if logger.isEnabledFor(logging.INFO): # Check if in non-verbose (INFO enabled)
         print(f"Processing input source: {input_source}", flush=True)
    else:
         logger.info(f"Processing input source: {input_source}") # Use logger info in verbose


    overall_pbar = None # Initialize overall progress bar to None

    if input_source.lower().startswith(('http://', 'https://')):
        # Input source type message - use print/tqdm.write in non-verbose
        if logger.isEnabledFor(logging.INFO):
            print("Input source type: URL", flush=True)
        else:
            logger.info("Input source type: URL") # Use logger info in verbose

        # Pass None for overall_pbar to download function - download has its own tqdm bar
        downloaded_file_path = download_file_from_url(input_source, overall_pbar=None)
        if downloaded_file_path:
            ext = os.path.splitext(downloaded_file_path)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                 # Fetch model only if we have a valid file to process and xAI key is present
                 if xai_api_key:
                     selected_xai_model = get_latest_xai_model()
                 # Process single downloaded file - no overall bar here
                 process_single_file(downloaded_file_path, assemblyai_api_key, xai_api_key, selected_xai_model, expected_speakers, force_transcribe, force_summarize, max_tokens, overall_pbar=None)
            else:
                 logger.warning(f"Downloaded file format '{ext}' is not explicitly listed as supported by AssemblyAI. Skipping processing.")
                 logger.debug(f"Supported extensions include: {', '.join(sorted(SUPPORTED_EXTENSIONS))}") # Debug only
                 try:
                     os.remove(downloaded_file_path)
                     logger.debug(f"Cleaned up unsupported downloaded file: {downloaded_file_path}") # Debug only
                 except OSError as e:
                     logger.warning(f"Error removing downloaded file {downloaded_file_path}: {e}")

    elif os.path.isdir(input_source):
        # Input source type message - use print/tqdm.write in non-verbose
        if logger.isEnabledFor(logging.INFO):
             print("Input source type: Folder", flush=True)
        else:
             logger.info("Input source type: Folder") # Use logger info in verbose

        supported_files_in_folder = []
        unsupported_files_in_folder = []
        total_duration = 0

        # Scanning folder message - use print/tqdm.write in non-verbose
        if logger.isEnabledFor(logging.INFO):
             print("Scanning folder for supported files and calculating total duration...", flush=True)
        else:
             logger.info("Scanning folder for supported files and calculating total duration...") # Use logger info in verbose


        for root, _, files in os.walk(input_source):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file_path)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    supported_files_in_folder.append(file_path)
                    duration = get_audio_duration(file_path)
                    if duration is not None:
                        total_duration += duration
                        logger.debug(f"Added duration {duration:.2f}s for {os.path.basename(file_path)}. Current total: {total_duration:.2f}s") # Debug only
                    else:
                        logger.debug(f"Could not get duration for {os.path.basename(file_path)} upfront using mutagen/tinytag. Progress for this file will be best-effort.") # Debug only
                else:
                     unsupported_files_in_folder.append(file_path)

        if unsupported_files_in_folder:
             logger.warning(f"\nSkipping {len(unsupported_files_in_folder)} files in the folder due to unsupported extensions:") # Keep warning
             for uf in unsupported_files_in_folder:
                  logger.warning(f"- {os.path.basename(uf)}") # Keep warning
             logger.warning("-" * 20) # Keep warning

        if not supported_files_in_folder:
            logger.warning(f"No supported audio/video files found in the folder '{input_source}' or its subdirectories.") # Keep warning
        else:
            # Found files message - use print/tqdm.write in non-verbose
            if logger.isEnabledFor(logging.INFO):
                 print(f"Found {len(supported_files_in_folder)} supported files in '{input_source}'.", flush=True)
            else:
                 logger.info(f"Found {len(supported_files_in_folder)} supported files in '{input_source}'.") # Use logger info in verbose


            if total_duration > 0 and logger.isEnabledFor(logging.INFO): # Check INFO level for pbar activation
                # Total duration message - use print/tqdm.write in non-verbose
                if logger.isEnabledFor(logging.INFO):
                     print(f"Total detected audio duration: {format_duration(total_duration)}", flush=True)
                else:
                     logger.info(f"Total detected audio duration: {format_duration(total_duration)}") # Use logger info in verbose

            # Fetch xAI model once before processing folder if xAI key is present
            if xai_api_key:
                selected_xai_model = get_latest_xai_model()

                # Initialize the overall progress bar - disabled if logger level is above INFO
                # Changed unit_scale to False to display raw seconds
                overall_pbar = tqdm(
                    total=total_duration,
                    unit='s',
                    unit_scale=False,
                    desc="Overall Progress",
                    leave=True,
                    disable=not logger.isEnabledFor(logging.INFO),
                    file=sys.stdout
                )

                # Update the custom logging handler with the active progress bar
                for handler in logger.handlers:
                    if isinstance(handler, TqdmWarningErrorHandler):
                        handler.overall_pbar = overall_pbar
                        break

            else:
                overall_pbar = None
                if total_duration == 0 and logger.isEnabledFor(logging.WARNING): # Check WARNING level for message
                     logger.warning("Could not detect duration for any supported files upfront. Overall duration progress bar disabled.") # Keep warning

            if supported_files_in_folder:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for i, file_path in enumerate(supported_files_in_folder):
                        futures.append(executor.submit(
                            process_single_file,
                            file_path=file_path,
                            assemblyai_api_key=assemblyai_api_key, # Use fetched key
                            xai_api_key=xai_api_key,             # Use fetched key
                            selected_xai_model=selected_xai_model, # Use fetched model
                            expected_speakers=expected_speakers,
                            force_transcribe=force_transcribe,
                            force_summarize=force_summarize,
                            max_tokens=max_tokens,
                            overall_pbar=overall_pbar,
                            file_index=i + 1,
                            total_files=len(supported_files_in_folder)
                        ))
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result() # Raise exceptions if any occurred in the thread
                        except Exception as exc:
                            # Log the exception. process_single_file does internal logging,
                            # but this catches issues during task submission or unexpected errors.
                            # Using overall_pbar.write ensures it's visible above the bar if active.
                            error_message = f"A file processing task generated an unhandled exception: {exc}"
                            if overall_pbar and not overall_pbar.disable:
                                overall_pbar.write(error_message)
                                overall_pbar.refresh()
                            else:
                                print(error_message, file=sys.stderr)
                            logger.error(error_message, exc_info=True) # Log with stack trace for details

            # Clear any potential leftover dynamic status line before closing bar
            if overall_pbar is not None and not overall_pbar.disable:
                 print(" " * 80, end='\r', flush=True)


            if overall_pbar:
                 overall_pbar.close()
                 print("", flush=True) # Ensure newline after bar

            # Reset the overall_pbar in the handler after processing all files
            for handler in logger.handlers:
                 if isinstance(handler, TqdmWarningErrorHandler):
                      handler.overall_pbar = None
                      break


    elif os.path.isfile(input_source):
        # Input source type message - use print/tqdm.write in non-verbose
        if logger.isEnabledFor(logging.INFO):
             print("Input source type: Local File", flush=True)
        else:
             logger.info("Input source type: Local File") # Use logger info in verbose

        ext = os.path.splitext(input_source)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            # Fetch model only if we have a valid file to process and xAI key is present
            if xai_api_key:
                selected_xai_model = get_latest_xai_model()
            process_single_file(input_source, assemblyai_api_key, xai_api_key, selected_xai_model, expected_speakers, force_transcribe, force_summarize, max_tokens, overall_pbar=None)
        else:
            logger.warning(f"Local file format '{ext}' is not explicitly listed as supported by AssemblyAI. Skipping processing.") # Keep warning
            logger.debug(f"Supported extensions include: {', '.join(sorted(SUPPORTED_EXTENSIONS))}") # Debug only

    else:
        logger.error(f"Error: Input source '{input_source}' is not a valid URL, file path, or folder path.") # Keep error
        parser.print_help()
        sys.exit(1)

    # Final processing complete message - use print/tqdm.write in non-verbose
    if logger.isEnabledFor(logging.INFO):
        print("Processing complete.", flush=True)
    else:
        logger.info("Processing complete.") # Use logger info in verbose
