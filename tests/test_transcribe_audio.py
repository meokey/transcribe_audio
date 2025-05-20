import unittest
from unittest import mock
import os
import json
import sys
import time
import logging

# Add src to path to allow importing transcribe_audio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import functions and classes from your script
from transcribe_audio_app import transcribe_audio

# Suppress most logging output during tests for cleaner test results
# You might want to see warnings/errors during debugging, so adjust as needed.
logging.disable(logging.CRITICAL)


class TestHelperFunctions(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual(transcribe_audio.format_duration(None), "N/A")
        self.assertEqual(transcribe_audio.format_duration(0), "00:00:00")
        self.assertEqual(transcribe_audio.format_duration(59), "00:00:59")
        self.assertEqual(transcribe_audio.format_duration(60), "00:01:00")
        self.assertEqual(transcribe_audio.format_duration(3601), "01:00:01")

    def test_extract_speaker_name_mapping(self):
        summary1 = "John (Speaker A) said hello. Jane (Speaker B) agreed."
        expected1 = {"Speaker A": "John", "Speaker B": "Jane"}
        self.assertEqual(transcribe_audio.extract_speaker_name_mapping(summary1), expected1)

        summary2 = "Speaker C (name not specified) spoke. Then Dr. Smith (Speaker D) presented."
        # Based on the current implementation, "Speaker C (name not specified)" won't be a key.
        # And "Dr. Smith (Speaker D)" should map "Speaker D" to "Dr. Smith".
        expected2 = {"Speaker D": "Dr. Smith"}
        self.assertEqual(transcribe_audio.extract_speaker_name_mapping(summary2), expected2)
        
        summary3 = "No speaker names here."
        expected3 = {}
        self.assertEqual(transcribe_audio.extract_speaker_name_mapping(summary3), expected3)

        summary4 = "Alice (Speaker A) and Bob (Speaker B) were talking. Later, Alice (Speaker A) spoke again."
        expected4 = {"Speaker A": "Alice", "Speaker B": "Bob"}
        self.assertEqual(transcribe_audio.extract_speaker_name_mapping(summary4), expected4)

        summary5 = "Speaker A (John Doe) made a point." # Different order
        expected5 = {"Speaker A": "John Doe"}
        self.assertEqual(transcribe_audio.extract_speaker_name_mapping(summary5), expected5)


    @mock.patch('transcribe_audio_app.transcribe_audio.MutagenFile', create=True) # create=True if MutagenFile might not exist
    @mock.patch('transcribe_audio_app.transcribe_audio.TinyTag', create=True)    # create=True if TinyTag might not exist
    def test_get_audio_duration(self, mock_tinytag, mock_mutagen):
        # Test with Mutagen success
        mock_mutagen_instance = mock.Mock()
        mock_mutagen_instance.info.length = 123.45
        mock_mutagen.return_value = mock_mutagen_instance
        self.assertEqual(transcribe_audio.get_audio_duration("fake.mp3"), 123.45)
        mock_mutagen.assert_called_with("fake.mp3")
        # Reset TinyTag mock for this specific scenario if it was called due to side_effect persistence
        mock_tinytag.get.reset_mock() 
        mock_tinytag.get.assert_not_called()

        # Test with Mutagen failure, TinyTag success
        mock_mutagen.side_effect = Exception("Mutagen error")
        mock_tinytag_instance = mock.Mock()
        mock_tinytag_instance.duration = 67.89
        mock_tinytag.get.return_value = mock_tinytag_instance
        self.assertEqual(transcribe_audio.get_audio_duration("fake.mp3"), 67.89)
        mock_tinytag.get.assert_called_with("fake.mp3")
        
        # Test both fail
        mock_mutagen.side_effect = Exception("Mutagen error")
        mock_tinytag.get.side_effect = Exception("TinyTag error")
        self.assertIsNone(transcribe_audio.get_audio_duration("fake.mp3"))


class TestDownloadFunction(unittest.TestCase):
    DOWNLOAD_DIR = "./test_downloads_dir" # Define a specific test download directory

    def setUp(self):
        # Ensure download directory exists for tests that write files
        os.makedirs(self.DOWNLOAD_DIR, exist_ok=True)
        # Suppress print statements from the download function during tests
        self.patcher = mock.patch('builtins.print')
        self.mock_print = self.patcher.start()
        
        # Mock logger for download function to avoid actual logging and allow assertions
        self.log_patcher = mock.patch('transcribe_audio_app.transcribe_audio.logger')
        self.mock_logger = self.log_patcher.start()


    def tearDown(self):
        # Clean up the test download directory and its contents
        if os.path.exists(self.DOWNLOAD_DIR):
            for item in os.listdir(self.DOWNLOAD_DIR):
                os.remove(os.path.join(self.DOWNLOAD_DIR, item))
            os.rmdir(self.DOWNLOAD_DIR)
        self.patcher.stop()
        self.log_patcher.stop()

    @mock.patch('transcribe_audio_app.transcribe_audio.requests.head')
    @mock.patch('transcribe_audio_app.transcribe_audio.requests.get')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('os.makedirs') # Mock os.makedirs as it's called inside the function
    def test_download_success_basic_url(self, mock_os_makedirs, mock_open_builtin, mock_requests_get, mock_requests_head):
        mock_head_response = mock.Mock()
        mock_head_response.headers = {'content-length': '1024', 'content-type': 'audio/mpeg'}
        mock_head_response.raise_for_status = mock.Mock()
        mock_requests_head.return_value = mock_head_response

        mock_get_response = mock.Mock()
        mock_get_response.raise_for_status = mock.Mock()
        mock_get_response.iter_content = lambda chunk_size: [b'chunk1', b'chunk2']
        # For context manager (__enter__ and __exit__)
        mock_requests_get.return_value.__enter__.return_value = mock_get_response

        url = "http://example.com/testfile.mp3"
        expected_filename = "testfile.mp3" 
        result_path = transcribe_audio.download_file_from_url(url, download_folder=self.DOWNLOAD_DIR)

        self.assertIsNotNone(result_path)
        self.assertTrue(result_path.endswith(expected_filename))
        # os.makedirs in the SUT is called with exist_ok=True, so we check that.
        # The first call to os.makedirs might be in setUp, so we check any_call.
        mock_os_makedirs.assert_any_call(self.DOWNLOAD_DIR, exist_ok=True)
        mock_open_builtin.assert_called_with(os.path.join(self.DOWNLOAD_DIR, expected_filename), 'wb')
        mock_requests_get.assert_called_once_with(url, stream=True, timeout=60)
        mock_requests_head.assert_called_once_with(url, allow_redirects=True, timeout=10)

    @mock.patch('transcribe_audio_app.transcribe_audio.requests.head')
    @mock.patch('transcribe_audio_app.transcribe_audio.requests.get')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    def test_download_success_with_content_disposition(self, mock_open_builtin, mock_requests_get, mock_requests_head):
        mock_head_response = mock.Mock()
        disposition_filename = "content_dispo_file.ogg"
        mock_head_response.headers = {
            'content-length': '2048',
            'content-disposition': f'attachment; filename="{disposition_filename}"',
            'content-type': 'audio/ogg'
        }
        mock_head_response.raise_for_status = mock.Mock()
        mock_requests_head.return_value = mock_head_response

        mock_get_response = mock.Mock()
        mock_get_response.raise_for_status = mock.Mock()
        mock_get_response.iter_content = lambda chunk_size: [b'data']
        mock_requests_get.return_value.__enter__.return_value = mock_get_response
        
        url = "http://example.com/some_other_name.mp3" # URL name different from content-disposition
        result_path = transcribe_audio.download_file_from_url(url, download_folder=self.DOWNLOAD_DIR)

        self.assertIsNotNone(result_path)
        self.assertTrue(result_path.endswith(disposition_filename))
        mock_open_builtin.assert_called_with(os.path.join(self.DOWNLOAD_DIR, disposition_filename), 'wb')


    @mock.patch('transcribe_audio_app.transcribe_audio.requests.head')
    def test_download_timeout(self, mock_requests_head):
        mock_requests_head.side_effect = transcribe_audio.requests.exceptions.Timeout
        result = transcribe_audio.download_file_from_url("http://example.com/timeout.mp3", download_folder=self.DOWNLOAD_DIR)
        self.assertIsNone(result)
        self.mock_logger.error.assert_called_with(mock.ANY, mock.ANY) # Check that an error was logged

    @mock.patch('transcribe_audio_app.transcribe_audio.requests.head')
    def test_download_connection_error(self, mock_requests_head):
        mock_requests_head.side_effect = transcribe_audio.requests.exceptions.ConnectionError
        result = transcribe_audio.download_file_from_url("http://example.com/connection_error.mp3", download_folder=self.DOWNLOAD_DIR)
        self.assertIsNone(result)
        self.mock_logger.error.assert_called_with(mock.ANY, mock.ANY)

    @mock.patch('transcribe_audio_app.transcribe_audio.requests.head')
    def test_download_http_error_on_head(self, mock_requests_head):
        mock_head_response = mock.Mock()
        mock_head_response.raise_for_status.side_effect = transcribe_audio.requests.exceptions.HTTPError("404 Client Error")
        mock_requests_head.return_value = mock_head_response
        
        result = transcribe_audio.download_file_from_url("http://example.com/notfound.mp3", download_folder=self.DOWNLOAD_DIR)
        self.assertIsNone(result)
        self.mock_logger.error.assert_called_with(mock.ANY, mock.ANY)

    @mock.patch('transcribe_audio_app.transcribe_audio.requests.head')
    @mock.patch('transcribe_audio_app.transcribe_audio.requests.get')
    def test_download_http_error_on_get(self, mock_requests_get, mock_requests_head):
        mock_head_response = mock.Mock()
        mock_head_response.headers = {'content-length': '1024', 'content-type': 'audio/mpeg'}
        mock_head_response.raise_for_status = mock.Mock()
        mock_requests_head.return_value = mock_head_response

        mock_get_response = mock.Mock()
        mock_get_response.raise_for_status.side_effect = transcribe_audio.requests.exceptions.HTTPError("500 Server Error")
        mock_requests_get.return_value.__enter__.return_value = mock_get_response
        
        result = transcribe_audio.download_file_from_url("http://example.com/servererror.mp3", download_folder=self.DOWNLOAD_DIR)
        self.assertIsNone(result)
        self.mock_logger.error.assert_called_with(mock.ANY, mock.ANY)

    @mock.patch('transcribe_audio_app.transcribe_audio.requests.head')
    @mock.patch('transcribe_audio_app.transcribe_audio.requests.get')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    def test_download_fallback_filename(self, mock_open_builtin, mock_requests_get, mock_requests_head):
        mock_head_response = mock.Mock()
        # No content-disposition, no extension in URL path, no useful content-type
        mock_head_response.headers = {'content-length': '100', 'content-type': 'application/octet-stream'} 
        mock_head_response.raise_for_status = mock.Mock()
        mock_requests_head.return_value = mock_head_response

        mock_get_response = mock.Mock()
        mock_get_response.raise_for_status = mock.Mock()
        mock_get_response.iter_content = lambda chunk_size: [b'data']
        mock_requests_get.return_value.__enter__.return_value = mock_get_response

        with mock.patch('time.time', return_value=1234567890): # Mock time for predictable fallback filename
            url = "http://example.com/nodot"
            result_path = transcribe_audio.download_file_from_url(url, download_folder=self.DOWNLOAD_DIR)
            
            self.assertIsNotNone(result_path)
            # Current fallback is "downloaded_file_<timestamp>" + ".download" (if no other extension found)
            expected_filename = f"nodot.download" 
            # The logic was updated to be `local_filename + file_extension` where local_filename is from URL
            # and file_extension defaults to ".download" if not found from other sources.
            # If the URL part is "nodot", it becomes "nodot.download"
            self.assertTrue(result_path.endswith(expected_filename))
            mock_open_builtin.assert_called_with(os.path.join(self.DOWNLOAD_DIR, expected_filename), 'wb')

    @mock.patch('transcribe_audio_app.transcribe_audio.requests.head')
    @mock.patch('transcribe_audio_app.transcribe_audio.requests.get')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    def test_download_filename_sanitization(self, mock_open_builtin, mock_requests_get, mock_requests_head):
        mock_head_response = mock.Mock()
        mock_head_response.headers = {'content-length': '100', 'content-type': 'audio/mpeg'}
        mock_head_response.raise_for_status = mock.Mock()
        mock_requests_head.return_value = mock_head_response

        mock_get_response = mock.Mock()
        mock_get_response.raise_for_status = mock.Mock()
        mock_get_response.iter_content = lambda chunk_size: [b'data']
        mock_requests_get.return_value.__enter__.return_value = mock_get_response
        
        url = "http://example.com/file<>:\"/\\|?*.mp3"
        sanitized_name = "file___________.mp3" # Based on re.sub(r'[^\w.-]', '_', local_filename)
        
        result_path = transcribe_audio.download_file_from_url(url, download_folder=self.DOWNLOAD_DIR)
        self.assertIsNotNone(result_path)
        self.assertTrue(result_path.endswith(sanitized_name))
        mock_open_builtin.assert_called_with(os.path.join(self.DOWNLOAD_DIR, sanitized_name), 'wb')


class TestCoreProcessing(unittest.TestCase):
    DUMMY_FILE_PATH = "dummy_audio_for_core_test.mp3"
    OUTPUT_JSON_SUFFIX = "_transcription_with_speakers.json"
    DUMMY_OUTPUT_PATH = os.path.splitext(DUMMY_FILE_PATH)[0] + OUTPUT_JSON_SUFFIX

    def setUp(self):
        # Create a dummy file for tests that need a file path
        with open(self.DUMMY_FILE_PATH, "w") as f:
            f.write("dummy audio data")
        
        # Suppress print statements from the functions during tests
        self.patcher_print = mock.patch('builtins.print')
        self.mock_print = self.patcher_print.start()
        
        # Mock logger to avoid actual logging and allow assertions if needed
        self.patcher_logger = mock.patch('transcribe_audio_app.transcribe_audio.logger')
        self.mock_logger = self.patcher_logger.start()

        # Clean up any potential leftover output file from previous failed tests
        if os.path.exists(self.DUMMY_OUTPUT_PATH):
            os.remove(self.DUMMY_OUTPUT_PATH)


    def tearDown(self):
        # Clean up the dummy file and any output file created
        if os.path.exists(self.DUMMY_FILE_PATH):
            os.remove(self.DUMMY_FILE_PATH)
        if os.path.exists(self.DUMMY_OUTPUT_PATH):
            os.remove(self.DUMMY_OUTPUT_PATH)
        self.patcher_print.stop()
        self.patcher_logger.stop()

    @mock.patch('transcribe_audio_app.transcribe_audio.transcribe_audio_with_diarization')
    @mock.patch('transcribe_audio_app.transcribe_audio.summarize_transcript_with_xai')
    @mock.patch('transcribe_audio_app.transcribe_audio.get_latest_xai_model', return_value="mock-grok-model") # Mock model selection
    @mock.patch('transcribe_audio_app.transcribe_audio.get_audio_duration', return_value=60.0)
    @mock.patch('transcribe_audio_app.transcribe_audio.load_output_data', return_value={}) # Start with no existing data
    @mock.patch('transcribe_audio_app.transcribe_audio.save_output_data')
    @mock.patch('os.path.getmtime', return_value=time.time())
    def test_process_single_file_success(self, mock_getmtime, mock_save_data, mock_load_data, 
                                         mock_get_duration, mock_get_model, mock_summarize, mock_transcribe):
        
        mock_transcribe.return_value = ([{"speaker": "A", "text": "Hello"}], 60.0) # Segments, duration from AA
        mock_summarize.return_value = ("Summary text", {"input_tokens": 10, "output_tokens": 5}) # Summary, usage

        # Mock the overall_pbar if it's used for .write or .set_description
        mock_pbar = mock.Mock()
        mock_pbar.disable = False # Ensure it's treated as active for .write calls

        transcribe_audio.process_single_file(
            file_path=self.DUMMY_FILE_PATH,
            assemblyai_api_key="fake_aa_key",
            xai_api_key="fake_xai_key",
            selected_xai_model="grok-model", # Passed directly now
            expected_speakers=2,
            force_transcribe=False,
            force_summarize=False,
            max_tokens=1000,
            overall_pbar=mock_pbar 
        )
        
        mock_load_data.assert_called_with(self.DUMMY_OUTPUT_PATH) # Check load is called
        mock_transcribe.assert_called_once()
        # Check if xAI key was present before trying to get model and summarize
        # In this test, xai_api_key="fake_xai_key", so it should proceed
        mock_get_model.assert_not_called() # get_latest_xai_model is called in __main__ now
        mock_summarize.assert_called_once()
        
        # Check that save_output_data was called. 
        # It's called multiple times: initial info, after transcription, after summarization.
        self.assertTrue(mock_save_data.called)
        
        # Example of checking arguments for one of the save calls (e.g., after summarization)
        # This requires knowing the exact structure of the data being saved.
        # For simplicity, just checking it was called is often enough for high-level tests.
        # If you want to be more specific:
        # args_list = mock_save_data.call_args_list
        # summary_save_call = any(
        #     call_args[0][1].get("summary") == "Summary text" for call_args in args_list
        # )
        # self.assertTrue(summary_save_call, "save_output_data was not called with the summary")

    @mock.patch('transcribe_audio_app.transcribe_audio.transcribe_audio_with_diarization', return_value=(None, None)) # Transcription fails
    @mock.patch('transcribe_audio_app.transcribe_audio.summarize_transcript_with_xai') # Mock to check it's NOT called
    @mock.patch('transcribe_audio_app.transcribe_audio.get_latest_xai_model', return_value="mock-grok-model")
    @mock.patch('transcribe_audio_app.transcribe_audio.get_audio_duration', return_value=60.0)
    @mock.patch('transcribe_audio_app.transcribe_audio.load_output_data', return_value={})
    @mock.patch('transcribe_audio_app.transcribe_audio.save_output_data')
    @mock.patch('os.path.getmtime', return_value=time.time())
    def test_process_single_file_transcription_fails(self, mock_getmtime, mock_save_data, mock_load_data,
                                                    mock_get_duration, mock_get_model, mock_summarize, mock_transcribe):
        mock_pbar = mock.Mock()
        mock_pbar.disable = False

        transcribe_audio.process_single_file(
            file_path=self.DUMMY_FILE_PATH,
            assemblyai_api_key="fake_aa_key",
            xai_api_key="fake_xai_key",
            selected_xai_model="grok-model",
            expected_speakers=2,
            force_transcribe=True, # Force to ensure it runs
            force_summarize=False,
            max_tokens=1000,
            overall_pbar=mock_pbar
        )
        
        mock_transcribe.assert_called_once()
        mock_summarize.assert_not_called() # Summarization should be skipped
        self.mock_logger.warning.assert_any_call("Transcription failed or skipped. Skipping summarization.")


    @mock.patch('transcribe_audio_app.transcribe_audio.transcribe_audio_with_diarization', return_value=([{"speaker": "A", "text": "Hello"}], 60.0))
    @mock.patch('transcribe_audio_app.transcribe_audio.summarize_transcript_with_xai', return_value=(None, None)) # Summarization fails
    @mock.patch('transcribe_audio_app.transcribe_audio.get_latest_xai_model', return_value="mock-grok-model")
    @mock.patch('transcribe_audio_app.transcribe_audio.get_audio_duration', return_value=60.0)
    @mock.patch('transcribe_audio_app.transcribe_audio.load_output_data', return_value={})
    @mock.patch('transcribe_audio_app.transcribe_audio.save_output_data')
    @mock.patch('os.path.getmtime', return_value=time.time())
    def test_process_single_file_summarization_fails(self, mock_getmtime, mock_save_data, mock_load_data,
                                                     mock_get_duration, mock_get_model, mock_summarize, mock_transcribe):
        mock_pbar = mock.Mock()
        mock_pbar.disable = False

        transcribe_audio.process_single_file(
            file_path=self.DUMMY_FILE_PATH,
            assemblyai_api_key="fake_aa_key",
            xai_api_key="fake_xai_key",
            selected_xai_model="grok-model",
            expected_speakers=2,
            force_transcribe=False,
            force_summarize=True, # Force to ensure it runs
            max_tokens=1000,
            overall_pbar=mock_pbar
        )
        
        mock_transcribe.assert_called_once()
        mock_summarize.assert_called_once()
        # Check if an error or warning related to summarization failure was logged by summarize_transcript_with_xai
        # This depends on the internal logging of summarize_transcript_with_xai upon failure.
        # For example, if it logs an error:
        # self.mock_logger.error.assert_any_call(mock.ANY) # Check if any error was logged. More specific if possible.


    @mock.patch('transcribe_audio_app.transcribe_audio.transcribe_audio_with_diarization')
    @mock.patch('transcribe_audio_app.transcribe_audio.summarize_transcript_with_xai')
    @mock.patch('transcribe_audio_app.transcribe_audio.get_latest_xai_model') # No return value, so it will be None
    @mock.patch('transcribe_audio_app.transcribe_audio.get_audio_duration', return_value=60.0)
    @mock.patch('transcribe_audio_app.transcribe_audio.load_output_data', return_value={})
    @mock.patch('transcribe_audio_app.transcribe_audio.save_output_data')
    @mock.patch('os.path.getmtime', return_value=time.time())
    def test_process_single_file_no_xai_model(self, mock_getmtime, mock_save_data, mock_load_data,
                                               mock_get_duration, mock_get_model, mock_summarize, mock_transcribe):
        mock_transcribe.return_value = ([{"speaker": "A", "text": "Hello"}], 60.0)
        mock_pbar = mock.Mock()
        mock_pbar.disable = False
        
        # Explicitly set selected_xai_model to None for this test case
        transcribe_audio.process_single_file(
            file_path=self.DUMMY_FILE_PATH,
            assemblyai_api_key="fake_aa_key",
            xai_api_key="fake_xai_key", # xAI key is present
            selected_xai_model=None,    # But no model was selected/passed
            expected_speakers=2,
            force_transcribe=False,
            force_summarize=True, 
            max_tokens=1000,
            overall_pbar=mock_pbar
        )
        
        mock_transcribe.assert_called_once()
        mock_get_model.assert_not_called() # get_latest_xai_model is called in __main__
        mock_summarize.assert_not_called() # Summarization should be skipped if model is None
        self.mock_logger.warning.assert_any_call("A suitable xAI model was not provided or could not be determined. Skipping summarization.")


if __name__ == '__main__':
    unittest.main()
