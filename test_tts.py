#!/usr/bin/env python3
"""
Test script for TTS functionality in Synthetic Memory Lite
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_memory import (
    get_available_voices,
    configure_tts,
    text_to_speech,
    clean_text_for_tts
)

def test_voice_selection():
    """Test voice selection functionality"""
    print("🔊 Testing Voice Selection...")
    
    voices = get_available_voices()
    print(f"✅ Found {len(voices)} available voices:")
    
    for voice_name, config in voices.items():
        print(f"  - {voice_name}: {config['lang']} ({config['tld']})")
    
    return True

def test_text_cleaning():
    """Test text cleaning for TTS"""
    print("\n🧹 Testing Text Cleaning...")
    
    test_text = """
    # Project Phoenix Feedback
    
    Based on the data provided:
    • **Project Phoenix** received positive feedback [1]
    • The architecture was praised as solid [2]
    
    Sources:
    • Email from sarah.chen@company.com: March 12
    • Slack message from alex_kim in #project-phoenix: March 18
    """
    
    cleaned = clean_text_for_tts(test_text)
    print(f"✅ Original text length: {len(test_text)}")
    print(f"✅ Cleaned text length: {len(cleaned)}")
    print(f"✅ Cleaned text preview: {cleaned[:100]}...")
    
    return True

def test_tts_generation():
    """Test TTS audio generation"""
    print("\n🎵 Testing TTS Generation...")
    
    try:
        # Test with a simple text
        test_text = "Hello! This is a test of the text-to-speech functionality."
        
        # Test with default voice
        audio_data = text_to_speech(test_text)
        
        if audio_data:
            print("✅ TTS generation successful!")
            print(f"✅ Audio data length: {len(audio_data)} characters")
            print(f"✅ Audio format: {audio_data[:50]}...")
            return True
        else:
            print("❌ TTS generation failed - no audio data returned")
            return False
            
    except Exception as e:
        print(f"❌ TTS generation failed with error: {str(e)}")
        return False

def test_voice_variations():
    """Test different voice configurations"""
    print("\n🌍 Testing Voice Variations...")
    
    voices = get_available_voices()
    test_text = "Testing different voice configurations."
    
    success_count = 0
    total_tests = min(3, len(voices))  # Test first 3 voices
    
    for i, (voice_name, config) in enumerate(list(voices.items())[:total_tests]):
        try:
            print(f"  Testing {voice_name}...")
            audio_data = text_to_speech(test_text, voice_name)
            
            if audio_data:
                print(f"    ✅ {voice_name} - Success")
                success_count += 1
            else:
                print(f"    ❌ {voice_name} - Failed")
                
        except Exception as e:
            print(f"    ❌ {voice_name} - Error: {str(e)}")
    
    print(f"✅ Voice variation test: {success_count}/{total_tests} successful")
    return success_count > 0

def main():
    """Run all TTS tests"""
    print("="*60)
    print("🧪 SYNTHETIC MEMORY LITE - TTS FUNCTIONALITY TEST")
    print("="*60)
    
    tests = [
        ("Voice Selection", test_voice_selection),
        ("Text Cleaning", test_text_cleaning),
        ("TTS Generation", test_tts_generation),
        ("Voice Variations", test_voice_variations)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} - ERROR: {str(e)}")
    
    print("\n" + "="*60)
    print("📊 TTS TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TTS TESTS PASSED!")
        print("🔊 Text-to-speech functionality is working correctly")
        return True
    else:
        print(f"\n⚠️  {failed} TTS TESTS FAILED")
        print("🔧 Please check the error messages above")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
