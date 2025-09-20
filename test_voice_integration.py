#!/usr/bin/env python3
"""
Voice Integration Test for Synthetic Memory Lite

This script tests the ElevenLabs voice integration functionality
without requiring the full Streamlit application.
"""

import os
import sys
import json
from unittest.mock import Mock, patch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_voice_functions():
    """Test voice-related functions"""
    print("🔊 Testing Voice Integration Functions...")
    
    try:
        from synthetic_memory import (
            extract_text_for_speech,
            configure_elevenlabs_api,
            get_available_voices,
            synthesize_speech
        )
        print("✅ Successfully imported voice functions")
    except ImportError as e:
        print(f"❌ Failed to import voice functions: {e}")
        return False
    
    # Test text extraction
    print("\n📝 Testing text extraction...")
    
    # Test with dictionary response
    dict_response = {
        "answer": "Project Phoenix received **positive feedback** [1] with some concerns [2].",
        "sources": [{"id": 1, "type": "email"}]
    }
    
    clean_text = extract_text_for_speech(dict_response)
    expected_clean = "Project Phoenix received positive feedback with some concerns."
    
    if clean_text == expected_clean:
        print("✅ Dictionary response text extraction works")
    else:
        print(f"❌ Dictionary response extraction failed. Got: '{clean_text}', Expected: '{expected_clean}'")
        return False
    
    # Test with string response
    string_response = "**Project Phoenix** received positive feedback from the team."
    clean_text = extract_text_for_speech(string_response)
    expected_clean = "Project Phoenix received positive feedback from the team."
    
    if clean_text == expected_clean:
        print("✅ String response text extraction works")
    else:
        print(f"❌ String response extraction failed. Got: '{clean_text}', Expected: '{expected_clean}'")
        return False
    
    # Test with complex text
    complex_text = """
    Based on the data provided:
    • Project Phoenix received positive feedback [1]
    • Technical approach was praised [2]
    
    Sources:
    • Email from sarah@company.com: March 12
    """
    
    clean_text = extract_text_for_speech(complex_text)
    if "Sources:" not in clean_text and "•" in clean_text:
        print("✅ Complex text extraction works")
    else:
        print(f"❌ Complex text extraction failed. Got: '{clean_text}'")
        return False
    
    print("\n🔧 Testing API configuration (mocked)...")
    
    # Mock secrets for testing
    with patch('synthetic_memory.st.secrets', {'ELEVENLABS_API_KEY': 'test-key-12345678901234567890'}):
        try:
            api_key = configure_elevenlabs_api()
            if api_key == 'test-key-12345678901234567890':
                print("✅ API configuration works with valid key")
            else:
                print(f"❌ API configuration failed. Got: '{api_key}'")
                return False
        except Exception as e:
            print(f"❌ API configuration failed: {e}")
            return False
    
    # Test with missing API key
    with patch('synthetic_memory.st.secrets', {}):
        try:
            configure_elevenlabs_api()
            print("❌ Should have failed with missing API key")
            return False
        except ValueError as e:
            if "ELEVENLABS_API_KEY not found" in str(e):
                print("✅ Properly handles missing API key")
            else:
                print(f"❌ Wrong error for missing API key: {e}")
                return False
    
    print("\n🎤 Testing voice synthesis (mocked)...")
    
    # Mock the requests.post call
    mock_response = Mock()
    mock_response.content = b"fake-audio-data"
    mock_response.raise_for_status.return_value = None
    
    with patch('requests.post', return_value=mock_response):
        try:
            audio_data = synthesize_speech(
                "Test text",
                "test-voice-id",
                "test-api-key",
                0.5,
                0.75
            )
            if audio_data == b"fake-audio-data":
                print("✅ Voice synthesis works with mocked API")
            else:
                print(f"❌ Voice synthesis failed. Got: {audio_data}")
                return False
        except Exception as e:
            print(f"❌ Voice synthesis failed: {e}")
            return False
    
    print("\n🎯 Testing voice fetching (mocked)...")
    
    # Mock the requests.get call for voice fetching
    mock_voices_response = Mock()
    mock_voices_response.json.return_value = {
        "voices": [
            {
                "voice_id": "test-voice-1",
                "name": "Test Voice",
                "description": "A test voice",
                "labels": {"accent": "american", "gender": "male", "age": "middle_aged"}
            }
        ]
    }
    mock_voices_response.raise_for_status.return_value = None
    
    with patch('requests.get', return_value=mock_voices_response):
        try:
            voices = get_available_voices("test-api-key")
            if len(voices) == 1 and voices[0]["name"] == "Test Voice":
                print("✅ Voice fetching works with mocked API")
            else:
                print(f"❌ Voice fetching failed. Got: {voices}")
                return False
        except Exception as e:
            print(f"❌ Voice fetching failed: {e}")
            return False
    
    return True


def test_ui_components():
    """Test UI component functions"""
    print("\n🖥️ Testing UI Components...")
    
    try:
        from synthetic_memory import create_voice_settings_ui, display_audio_player
        print("✅ Successfully imported UI functions")
    except ImportError as e:
        print(f"❌ Failed to import UI functions: {e}")
        return False
    
    # Test display_audio_player with mock data
    try:
        display_audio_player(b"test-audio-data", "Test text")
        print("✅ Audio player display function works")
    except Exception as e:
        print(f"❌ Audio player display failed: {e}")
        return False
    
    return True


def main():
    """Run all voice integration tests"""
    print("="*60)
    print("🔊 SYNTHETIC MEMORY LITE - VOICE INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        ("Voice Functions", test_voice_functions),
        ("UI Components", test_ui_components)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "="*60)
    print("📊 VOICE INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL VOICE INTEGRATION TESTS PASSED!")
        print("🔊 Voice functionality is ready for use")
        return True
    else:
        print(f"\n⚠️  {failed} VOICE INTEGRATION TESTS FAILED")
        print("🔧 Please fix the issues before using voice features")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
