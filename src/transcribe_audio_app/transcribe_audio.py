import os
import assemblyai
import json
import argparse
from dotenv import load_dotenv
import getpass
import requests
import re  # Import regex module
import datetime  # Import datetime module
import tenacity  # Import tenacity for retries
import logging  # Import logging for before_sleep_log

# Removed 'from assemblyai import exceptions' as it caused ImportError


# --- Configuration ---
XAI_API_BASE_URL = "https://api.x.ai"  # Base URL for xAI API
# Updated endpoint for message-based interactions based on documentation
XAI_MESSAGES_ENDPOINT = f"{XAI_API_BASE_URL}/v1/messages"
XAI_MODELS_ENDPOINT = f"{XAI_API_BASE_URL}/v1/models"  # Endpoint to list models

TRANSCRIPT_FILENAME_SUFFIX = "_transcription_with_speakers.json"

# Configure logging (optional, but useful with before_sleep_log)
# logging.basicConfig(level=logging.INFO) # Uncomment to see retry logs
# Get a logger instance
logger = logging.getLogger(__name__)

# --- Retry Configuration ---
# Define retry settings with exponential backoff and jitter
# Retries up to 5 times, waiting exponentially between attempts (~1s, ~2s, ~4s, ~8s, ~16s + jitter)
# Max wait between retries is capped at 60 seconds
common_retry_settings = tenacity.retry(
    wait=tenacity.wait_exponential_jitter(
        max=60, jitter=10
    ),  # Wait base is 1s (default), max 60s, add up to 10s jitter
    stop=tenacity.stop_after_attempt(5),  # Retry up to 5 times
    before_sleep=tenacity.before_sleep_log(
        logger, logging.INFO
    ),  # Log retry message using logging
    reraise=True,  # Re-raise the exception if retries are exhausted
)

# Define specific condition for retrying HTTP errors (e.g., rate limits, server errors)
retry_if_transient_http_error = tenacity.retry_if_exception(
    lambda e: isinstance(e, requests.exceptions.HTTPError)
    and e.response.status_code in {429, 500, 502, 503, 504}
)

# Define retry decorator specifically for AssemblyAI API calls (Workaround)
assemblyai_retry_settings = tenacity.retry(
    wait=tenacity.wait_exponential_jitter(
        max=60, jitter=10
    ),  # Wait base is 1s (default), max 60s, add up to 10s jitter
    stop=tenacity.stop_after_attempt(5),
    # Workaround: Retry on general requests exceptions and transient HTTP errors
    # instead of AssemblyAI's specific ApiException which is causing import issues.
    retry=(
        tenacity.retry_if_exception_type(
            requests.exceptions.RequestException
        )  # Retry on general requests issues
        | retry_if_transient_http_error  # Retry on specific transient HTTP errors
    ),
    before_sleep=tenacity.before_sleep_log(
        logger, logging.INFO
    ),  # Log retry message using logging
    reraise=True,  # Re-raise if retries fail
)


# Removed call_assemblyai_transcribe helper - applying decorator directly to transcribe_audio_with_diarization's logic


def load_output_data(filepath):
    """Loads data from a JSON file, handling errors."""
    data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, dict):
                    data = loaded_data
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode existing JSON file '{filepath}'. Starting with an empty structure."
            )
        except Exception as e:
            print(
                f"Warning: Error reading existing JSON file '{filepath}': {e}. Starting with an empty structure."
            )
    return data


def save_output_data(filepath, data):
    """Saves data to a JSON file."""
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        # print(f"Data saved to '{filepath}'.") # Removed verbose save message
    except Exception as e:
        print(f"Error saving data to '{filepath}': {e}")


def get_assemblyai_api_key():
    """Gets the AssemblyAI API key from environment variables or asks the user with masking."""
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass(
            "Please enter your AssemblyAI API key for transcription: "
        )
    return api_key


def get_xai_api_key():
    """Gets the xAI API key from environment variables or asks the user with masking."""
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Please enter your xAI API key for summarization: ")
    return api_key


def find_audio_file(filename):
    """Searches for the audio file in the current folder or ./data/ folder."""
    if os.path.exists(filename):
        return filename
    data_folder = "./data/"
    filepath = os.path.join(data_folder, filename)
    if os.path.exists(filepath):
        return filepath
    return None


# --- AssemblyAI Transcription Function with Retry Workaround ---
# Apply the retry decorator directly to the core transcription logic within the function
def transcribe_audio_with_diarization(
    audio_file_path, assemblyai_api_key, num_speakers=None, force=False
):
    """
    Transcribes a local MP3 audio file with speaker diarization using AssemblyAI.

    Args:
        audio_file_path (str): The path to the local MP3 audio file.
        assemblyai_api_key (str): The AssemblyAI API key.
        num_speakers (int, optional): The expected number of speakers.
                                       Providing this can improve accuracy. Defaults to None.
        force (bool, optional): Force transcription even if transcript data exists in output file. Defaults to False.

    Returns:
        list: A list of dictionaries, where each dictionary represents a spoken segment
              and contains the speaker label and the transcribed text.
              Returns None if transcription fails or is skipped.
    """
    output_filename = os.path.splitext(audio_file_path)[0] + TRANSCRIPT_FILENAME_SUFFIX
    output_data = load_output_data(output_filename)

    if "transcript" in output_data and not force:
        print(
            f"Transcript data found in '{output_filename}'. Skipping transcription (use --force-transcribe to re-run)."
        )
        # Return the segments list directly if it exists
        return output_data.get("transcript", {}).get("segments")

    if not assemblyai_api_key:
        print("Error: AssemblyAI API key not provided. Exiting transcription.")
        return None

    assemblyai.settings.api_key = assemblyai_api_key  # Use the passed key

    print(f"Starting transcription for '{audio_file_path}'...")
    try:
        # Define the retryable core logic as a nested function or apply decorator here if possible
        @assemblyai_retry_settings  # Apply the retry decorator here
        def _transcribe_core(file_path, num_speakers_expected):
            config = assemblyai.TranscriptionConfig(
                speaker_labels=True,
                speakers_expected=num_speakers_expected,  # Use the passed arg
            )
            transcriber = assemblyai.Transcriber(config=config)
            return transcriber.transcribe(file_path)  # This call is now retried

        # Call the retryable core logic
        transcript = _transcribe_core(audio_file_path, num_speakers)

        if transcript and transcript.utterances:
            segments = []
            for utterance in transcript.utterances:
                segments.append(
                    {"speaker": f"Speaker {utterance.speaker}", "text": utterance.text}
                )
            # Ensure 'transcript' key exists before adding segments
            if "transcript" not in output_data:
                output_data["transcript"] = {}
            output_data["transcript"]["segments"] = segments

            save_output_data(output_filename, output_data)
            print(f"Transcription complete and saved to '{output_filename}'.")
            return segments
        else:
            print("Transcription failed or no utterances found.")
            return None

    except tenacity.RetryError as e:
        # This exception is raised by tenacity if all retries fail
        print(f"Failed to transcribe audio after multiple retries: {e}")
        # Attempt to print details from the last attempt if it was a requests HTTP error
        if e.cause and isinstance(e.cause, requests.exceptions.RequestException):
            if hasattr(e.cause, "response") and hasattr(
                e.cause.response, "status_code"
            ):
                print(
                    f"Last attempt failed with HTTP status: {e.cause.response.status_code}"
                )
                try:
                    error_details = e.cause.response.json()
                    print(f"Error details: {error_details}")
                except json.JSONDecodeError:
                    print(f"Error response body: {e.cause.response.text}")
            else:
                print(
                    f"Last attempt failed with request exception: {e.cause}"
                )  # Print other request exception details
        return None
    # Catch requests exceptions that weren't retried (e.g., 400, 401, 404) or other non-retryable request issues
    except requests.exceptions.RequestException as e:
        print(
            f"A requests error occurred during AssemblyAI transcription that was not retried: {e}"
        )
        # Print specific HTTP error details if available
        if isinstance(e, requests.exceptions.HTTPError):
            print(f"HTTP Status: {e.response.status_code}")
            try:
                print(f"Response Body: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Response Body (non-JSON): {e.response.text}")
        return None
    except Exception as e:  # Catch any other unexpected errors
        print(f"An unexpected error occurred during transcription: {e}")
        return None


# --- Retryable xAI Models List Call ---
@common_retry_settings
@tenacity.retry(
    retry=retry_if_transient_http_error
    | tenacity.retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_xai_models_with_retry(api_key):
    """Fetches xAI models with retry logic."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.get(XAI_MODELS_ENDPOINT, headers=headers)
    response.raise_for_status()  # Raise HTTPError for bad status codes (handled by retry_if_transient_http_error)
    return response.json()


def get_latest_xai_model(api_key):
    """
    Fetches the list of available xAI models, prints the list, and selects a model
    based on the prioritized grok pattern (grok-[num]-mini-fast-beta, grok-[num]-beta, grok-[num])
    with the highest number, falling back to "grok-3-latest" if available, then the
    first available model if no other option is suitable.

    Args:
        api_key (str): The xAI API key.

    Returns:
        str: The selected model ID, or None if fetching fails or no suitable model found.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    print("Attempting to fetch list of available xAI models...")
    available_models = []
    try:
        # Call the retryable function
        models_data = fetch_xai_models_with_retry(api_key)

        if (
            not models_data
            or "data" not in models_data
            or not isinstance(models_data["data"], list)
        ):
            print("Error: Unexpected response format from xAI models endpoint.")
            return None

        available_models = [
            model["id"] for model in models_data["data"] if "id" in model
        ]

    except tenacity.RetryError as e:
        # This exception is raised by tenacity if all retries fail
        print(f"Failed to fetch xAI models after multiple retries: {e}")
        return None
    except Exception as e:  # Catch other potential errors during fetch or processing
        print(f"An error occurred while fetching or processing xAI models: {e}")
        return None

    # --- Print all available models ---
    if available_models:
        print("\nAvailable models from xAI:")
        for model_id in available_models:
            print(f"- {model_id}")
        print("-" * 20)
    else:
        print("\nNo models retrieved from xAI.")
        return None  # Return None if the list is empty

    # --- Model Selection Logic based on prioritized grok patterns ---
    grok_pattern = re.compile(r"^grok-(\d+)(?:-.*)?$")
    grok_models_with_numbers = []

    for model_id in available_models:
        match = grok_pattern.match(model_id)
        if match:
            number = int(match.group(1))
            grok_models_with_numbers.append((number, model_id))

    selected_model = None

    if grok_models_with_numbers:
        # Find the highest number among grok models
        highest_number = max([num for num, _ in grok_models_with_numbers])

        # Construct NEW prioritized model names using the highest number
        prioritized_patterns = [
            f"grok-{highest_number}-mini-fast-beta",  # New pattern
            f"grok-{highest_number}-beta",
            f"grok-{highest_number}",
        ]

        print(
            f"Highest grok version found: {highest_number}. Checking prioritized patterns:"
        )
        for pattern in prioritized_patterns:
            if pattern in available_models:
                selected_model = pattern
                print(f"Selected model matching pattern: {selected_model}")
                return selected_model  # Return the first match

        # If none of the patterns with the highest number are found, fall through
        print(f"None of the prioritized grok-{highest_number} patterns found.")

    else:
        print("No models matching the 'grok-N' pattern found.")

    # --- Final Fallback: Check for "grok-3-latest" then use the very first model ---
    # This is reached if no grok models matching the pattern were found,
    # or if grok models were found but none of the prioritized patterns
    # with the highest number matched.

    fallback_specific = "grok-3-latest"
    if fallback_specific in available_models:
        selected_model = fallback_specific
        print(f"Falling back to specific model: {selected_model}")
        return selected_model  # Return the specific fallback

    # Final final fallback: Use the very first model in the available list
    # available_models is guaranteed not to be empty here if we reached this point
    # and didn't return None from the initial check.
    selected_model = available_models[0]
    print(
        f"Neither prioritized grok patterns nor '{fallback_specific}' found. Falling back to selecting the first available model: {selected_model}"
    )
    return selected_model


def extract_speaker_name_mapping(summary_text):
    """
    Analyzes the summary text to find potential speaker name mappings
    like "Name (Speaker X)" or "Speaker X (Name)".

    Args:
        summary_text (str): The generated summary string.

    Returns:
        dict: A dictionary mapping original speaker labels to identified names
              (e.g., {"Speaker A": "John", "Speaker B": "Sarah"}).
        Note: This is a basic extraction and may not catch all name attributions.
    """
    if not summary_text:
        return {}

    name_mapping = {}

    # Pattern 1: "Name (Speaker X)" - e.g., John (Speaker A)
    # Catches single names or multi-word names separated by spaces
    pattern1 = re.compile(
        r"(\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+)*|\b[A-Z]+\b)\s+\(Speaker ([A-Z])\)"
    )
    matches1 = pattern1.findall(summary_text)
    for name, speaker_label_suffix in matches1:
        original_label = f"Speaker {speaker_label_suffix}"
        # Clean up name (remove leading/trailing whitespace)
        name_mapping[original_label] = name.strip()

    # Pattern 2: "Speaker X (Name)" - e.g., Speaker C (Michael)
    # Catches single names or multi-word names separated by spaces
    pattern2 = re.compile(
        r"Speaker ([A-Z])\s+\((\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+)*|\b[A-Z]+\b)\)"
    )
    matches2 = pattern2.findall(summary_text)
    for speaker_label_suffix, name in matches2:
        original_label = f"Speaker {speaker_label_suffix}"
        # Clean up name
        name_mapping[original_label] = name.strip()

    # Note: This basic parsing might miss names used without the Speaker X label in the summary
    # or names used in conversational context without a direct mapping provided by the LLM.

    return name_mapping


# --- Retryable xAI Messages Post Call ---
@common_retry_settings
@tenacity.retry(
    retry=retry_if_transient_http_error
    | tenacity.retry_if_exception_type(requests.exceptions.RequestException)
)
def post_xai_messages_with_retry(api_key, model_id, prompt_text, max_tokens):
    """Posts message to xAI API with retry logic."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,  # Use the dynamically determined model ID
        "messages": [
            {"role": "user", "content": prompt_text}  # Use the updated prompt
        ],
        "max_tokens": max_tokens,  # Use the passed max_tokens value
        # Add other parameters as needed based on xAI API documentation for /v1/messages
    }

    response = requests.post(XAI_MESSAGES_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()  # Raise HTTPError (handled by retry_if_transient_http_error)
    return response.json()


def summarize_transcript_with_xai(
    transcript_segments,
    audio_file_path,
    xai_api_key,
    model_id,
    audio_datetime_str=None,
    force=False,
    max_tokens=10000,
):
    """
    Summarizes the given transcript segments using a specified xAI LLM model,
    extracts identified speaker names from the summary to update the transcript,
    and saves the summary, usage data, and updated transcript to the output file.

    Args:
        transcript_segments (list): A list of dictionaries representing the transcribed audio segments.
        audio_file_path (str): The path to the local MP3 audio file (for output filename).
        xai_api_key (str): The xAI API key.
        model_id (str): The ID of the xAI model to use for summarization.
        audio_datetime_str (str, optional): Formatted date/time string of the audio recording. Defaults to None.
        force (bool, optional): Force summarization even if summary exists in output file. Defaults to False.
        max_tokens (int, optional): Maximum number of tokens for the xAI response. Defaults to 10000.

    Returns:
        str: The summary generated by xAI, or None if summarization fails or is skipped.
    """
    output_filename = os.path.splitext(audio_file_path)[0] + TRANSCRIPT_FILENAME_SUFFIX
    # Load output_data at the start to ensure we have the latest transcript segments
    # and other info like audio_info before adding summary/usage or updating segments.
    output_data = load_output_data(output_filename)

    # Check if summary or xai_usage exists and force is not requested
    if ("summary" in output_data or "xai_usage" in output_data) and not force:
        print(
            f"Summary or xAI usage data found in '{output_filename}'. Skipping summarization (use --force-summarize to re-run)."
        )
        # Return the summary if it exists, otherwise None
        return output_data.get("summary")

    if not xai_api_key:
        print("Error: xAI API key not provided. Exiting summarization.")
        return None

    if not XAI_MESSAGES_ENDPOINT:
        print("Error: xAI Messages API endpoint not configured. Exiting summarization.")
        return None

    # Format the transcript for the prompt
    full_transcript_text = "\n".join(
        [f"{item['speaker']}: {item['text']}" for item in transcript_segments]
    )

    headers = {
        "Authorization": f"Bearer {xai_api_key}",
        "Content-Type": "application/json",
    }

    # --- Prompt: Instructing the model on Markdown, meeting format, date, and NEW speaker naming rules ---
    # Includes instructions for Markdown, bullet points, meeting format, date,
    # AND using identified speaker names instead of labels, with first vs subsequent mention formatting.

    prompt_text = f"""Summarize the following conversation transcript. Format the output in Markdown.

Identify speakers by their names from the transcript content whenever possible. If a speaker's name cannot be determined from the transcript content, use their assigned label (e.g., Speaker A, Speaker B) from the transcript.

When you first mention a speaker in the summary (either by name or label), format it as follows:
- If the name is known: "Name (Speaker X)" (e.g., "Bill (Speaker A)")
- If the name is unknown: "Speaker X (name not specified)" (e.g., "Speaker B (name not specified)")
For all *subsequent* mentions of that same speaker in the summary, use only their identified name or their label (e.g., just "Bill" or just "Speaker A"), without any additional comments in brackets.

Focus on key discussion points, using a bulleted list (`- `). Attribute points to speakers using their name or label as described above (applying the first mention vs. subsequent mention rule).

If the conversation resembles a meeting, include relevant sections formatted as Markdown, such as:
- **Date:** {audio_datetime_str if audio_datetime_str else 'N/A'}
- **Attendees:** [List identified speakers by name, or label if name is unknown. Use the first mention format defined above for each attendee listed here.]
- **Topics Discussed:** (Reference the bulleted list of key points)
- **Decisions:** [List any decisions identified]
- **Action Items:** [List any action items identified]

If it is not a meeting, provide a concise summary using a Markdown bulleted list for key topics.

Transcript:
{full_transcript_text}
"""

    # Payload structure for the /v1/messages endpoint
    payload = {
        "model": model_id,  # Use the dynamically determined model ID
        "messages": [
            {"role": "user", "content": prompt_text}  # Use the updated prompt
        ],
        "max_tokens": max_tokens,  # Use the passed max_tokens value
        # Add other parameters as needed based on xAI API documentation for /v1/messages
    }

    print(
        f"Sending summarization request to xAI using model: {model_id} with max_tokens={max_tokens}..."
    )
    try:
        # Call the retryable function
        summary_data = post_xai_messages_with_retry(
            xai_api_key, model_id, prompt_text, max_tokens
        )

        # --- Extract summary and usage based on the provided response structure ---
        summary = None
        usage = None

        # Extract summary text from the content list (as per your error output structure)
        if (
            summary_data
            and "content" in summary_data
            and isinstance(summary_data["content"], list)
        ):
            # Iterate through content blocks to find the text block
            for block in summary_data["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and "text" in block
                ):
                    summary = block["text"]
                    break  # Found the text summary, stop searching

        # Extract usage information
        if (
            summary_data
            and "usage" in summary_data
            and isinstance(summary_data["usage"], dict)
        ):
            usage = summary_data["usage"]
        # --- End Extraction ---

        if summary:
            # Ensure 'summary' key is updated in output_data
            output_data["summary"] = summary
            print(f"Summary generated successfully.")

            # Save usage data if available
            if usage is not None:
                output_data["xai_usage"] = usage
                print(f"xAI usage data recorded: {usage}")

            # --- Extract speaker name mapping and update transcript segments ---
            print(
                "Attempting to extract speaker names from summary for transcript update..."
            )
            speaker_name_mapping = extract_speaker_name_mapping(summary)

            if speaker_name_mapping:
                print(f"Identified speaker names to map: {speaker_name_mapping}")
                # Access the transcript segments that are already loaded in output_data
                # Ensure segments exist in the loaded data before attempting update
                if (
                    "transcript" in output_data
                    and isinstance(output_data["transcript"], dict)
                    and "segments" in output_data["transcript"]
                    and isinstance(output_data["transcript"]["segments"], list)
                ):
                    print("Updating transcript segments with identified names...")
                    updated_segments = []
                    # Iterate through a copy of the segments list to avoid issues while potentially modifying
                    for segment in output_data["transcript"]["segments"].copy():
                        original_speaker_label = segment.get("speaker")
                        # Check if the original label exists in the mapping and if segment has a speaker key
                        if (
                            original_speaker_label
                            and original_speaker_label in speaker_name_mapping
                        ):
                            # Create a new segment dictionary with the updated speaker name
                            updated_segment = segment.copy()
                            updated_segment["speaker"] = speaker_name_mapping[
                                original_speaker_label
                            ]
                            updated_segments.append(updated_segment)
                        else:
                            updated_segments.append(
                                segment
                            )  # Keep original segment if no mapping or no speaker key

                    # Replace the old segments list in output_data with the new one
                    output_data["transcript"]["segments"] = updated_segments
                    print("Transcript segments updated.")
                else:
                    print(
                        "Warning: Transcript segments not found in output_data in expected format for name mapping update."
                    )
            else:
                print(
                    "No speaker names identified from the summary for transcript update."
                )
            # --- End New ---

            # Save the updated output_data (includes summary, usage, audio_info, and potentially updated transcript)
            save_output_data(output_filename, output_data)
            print(
                f"Summary, usage, audio info, and updated transcript saved to '{output_filename}'."
            )

            return summary
        else:
            print(
                f"Error: Could not extract summary text from xAI API response. Response data: {summary_data}"
            )
            # Still attempt to save usage data if available, even if summary extraction failed
            # Note: transcript name mapping is skipped if no summary text is extracted
            if usage is not None:
                output_data["xai_usage"] = usage
                print(
                    f"xAI usage data recorded despite summary extraction failure: {usage}"
                )
                save_output_data(
                    output_filename, output_data
                )  # Save again to include usage
                print(f"Usage data saved to '{output_filename}'.")

            return None

    except tenacity.RetryError as e:
        # This exception is raised by tenacity if all retries fail
        print(f"Failed to generate summary after multiple retries: {e}")
        return None
    except Exception as e:  # Catch other potential errors during post or processing
        print(f"An error occurred during summarization: {e}")
        return None


if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file

    parser = argparse.ArgumentParser(
        description="Transcribe audio and summarize using AssemblyAI and xAI."
    )
    parser.add_argument("audio_file", help="Name of the local MP3 audio file.")
    parser.add_argument(
        "-s",
        "--speakers",
        type=int,
        default=3,
        help="Expected number of speakers (optional, default is 3).",
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
    # Add max_tokens argument
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum number of tokens for the xAI summarization model (default: 10000).",
    )
    args = parser.parse_args()

    audio_filename = args.audio_file
    expected_speakers = args.speakers
    force_transcribe = args.force_transcribe
    force_summarize = args.force_summarize
    max_tokens = args.max_tokens  # Get the value

    audio_file_path = find_audio_file(audio_filename)

    if not audio_file_path:
        print(f"Error: Audio file '{audio_filename}' not found.")
        exit(1)  # Use non-zero exit code for errors

    if (
        os.path.getsize(audio_file_path) > 2 * 60 * 60 * 1024 * 1024
    ):  # 2 hours limit in bytes
        print("Error: Audio file size exceeds 2 hour limit (approx 2GB).")
        print("Consider splitting the audio into smaller chunks.")
        exit(1)  # Use non-zero exit code for errors

    output_filename = os.path.splitext(audio_file_path)[0] + TRANSCRIPT_FILENAME_SUFFIX
    output_data = load_output_data(output_filename)  # Load data at the start

    # --- Get Audio Date/Time ---
    audio_datetime_str = None
    try:
        # Using modification time as a proxy for recording date/time
        mtime = os.path.getmtime(audio_file_path)
        audio_datetime_str = datetime.datetime.fromtimestamp(mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )  # Format date/time
        print(f"Detected audio file date/time: {audio_datetime_str}")
        # Store date/time in output_data if not already present or if forcing
        if "audio_info" not in output_data or force_transcribe or force_summarize:
            output_data["audio_info"] = {"recorded_datetime": audio_datetime_str}
            # Save immediately after adding date, in case later steps fail
            save_output_data(output_filename, output_data)
            print(f"Audio info saved to '{output_filename}'.")

    except OSError as e:
        print(
            f"Warning: Could not get file modification time for '{audio_file_path}': {e}"
        )
    except Exception as e:
        print(
            f"Warning: An unexpected error occurred while getting audio date/time: {e}"
        )

    # --- Get API Keys once ---
    assemblyai_api_key = get_assemblyai_api_key()
    xai_api_key = get_xai_api_key()

    # --- Transcription ---
    # Pass the key to the transcription function
    # transcribe updates output_data and saves.
    # The function returns the segments list (either new or loaded)
    transcript_segments = transcribe_audio_with_diarization(
        audio_file_path, assemblyai_api_key, expected_speakers, force=force_transcribe
    )

    # --- Summarization ---
    # summarize receives the transcript_segments returned by transcribe
    # It will load the latest output_data internally (which includes audio_info and potentially existing transcript)
    # before adding summary/usage and updating the segments with names
    if transcript_segments:
        if xai_api_key:
            # Get the model dynamically with updated preference patterns and list all models
            selected_model = get_latest_xai_model(xai_api_key)
            if selected_model:
                # Pass the key, the selected model, audio_datetime_str, AND max_tokens to summarization
                summarize_transcript_with_xai(
                    transcript_segments,
                    audio_file_path,
                    xai_api_key,
                    selected_model,
                    audio_datetime_str=audio_datetime_str,
                    force=force_summarize,
                    max_tokens=max_tokens,
                )
            else:
                print(
                    "Could not determine a suitable xAI model from the available list. Skipping summarization."
                )
        else:
            print("xAI API key not available. Skipping summarization.")
    else:
        print("Transcription failed. Skipping summarization.")
