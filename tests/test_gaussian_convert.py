import pytest
from unittest.mock import patch, MagicMock
import sys
import os

def test_convert_script_subprocess_calls():
    # Mocking sys.argv to provide required arguments
    test_args = [
        "convert.py",
        "--source_path", "/tmp/source",
        "--colmap_executable", "colmap",
        "--magick_executable", "magick",
        "--resize"
    ]
    
    # Mocking os and subprocess
    with patch.object(sys, 'argv', test_args), \
         patch("subprocess.run") as mock_run, \
         patch("os.makedirs"), \
         patch("os.listdir", return_value=["img1.jpg"]), \
         patch("shutil.move"), \
         patch("shutil.copy2"):
        
        # We need to use runpy or importlib to execute the script
        import runpy
        runpy.run_path("/home/elix3r/projects/TalkingHeadAvatar/GaussianAvatars/convert.py")
        
        # Verify subprocess.run calls
        assert mock_run.called
        for call in mock_run.call_args_list:
            args_list = call[0][0]
            # Assert that it's a list of strings
            assert isinstance(args_list, list)
            assert all(isinstance(arg, str) for arg in args_list)
            # Assert that check=True is passed
            assert call[1].get("check") is True

def test_convert_script_shell_injection_protection():
    # Malicious file name that would normally trigger command injection if using os.system with strings
    malicious_file = "'; touch /tmp/pwned; '.jpg"
    
    test_args = [
        "convert.py",
        "--source_path", "/tmp/source",
        "--resize"
    ]
    
    with patch.object(sys, 'argv', test_args), \
         patch("subprocess.run") as mock_run, \
         patch("os.makedirs"), \
         patch("os.listdir", side_effect=[["sparse_subdir"], [malicious_file]]), \
         patch("shutil.move"), \
         patch("shutil.copy2"), \
         patch("os.path.join", side_effect=os.path.join):
        
        import runpy
        runpy.run_path("/home/elix3r/projects/TalkingHeadAvatar/GaussianAvatars/convert.py")
        
        # Check the mogrify calls
        mogrify_calls = [call for call in mock_run.call_args_list if "mogrify" in call[0][0]]
        for call in mogrify_calls:
            args_list = call[0][0]
            # The malicious file name should be a single element in the list, not split by shell
            assert any(malicious_file in arg for arg in args_list)
