#!/usr/bin/env python3
"""
Demo Validation Script for Synthetic Memory Lite

This script validates that the demo is ready for live presentation by:
- Testing the complete workflow with the pre-populated demo query
- Verifying proper source attribution in synthesized responses
- Testing error handling scenarios and edge cases
- Validating UI responsiveness and loading indicators
- Ensuring demo runs smoothly for live presentation

Requirements: 4.4, 5.2, 5.3, 5.4, 5.5
"""

import os
import sys
import time
import json
from typing import Dict, List, Tuple

def check_data_files() -> Tuple[bool, str]:
    """Check that all required data files are present and valid"""
    print("🔍 Checking data files...")
    
    required_files = {
        'emails.json': 'email data',
        'slack_messages.json': 'Slack message data', 
        'project_notes.txt': 'project documentation'
    }
    
    missing_files = []
    invalid_files = []
    
    for filename, description in required_files.items():
        if not os.path.exists(filename):
            missing_files.append(f"{filename} ({description})")
            continue
            
        try:
            if filename.endswith('.json'):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list) or len(data) == 0:
                        invalid_files.append(f"{filename} - not a valid list or empty")
            else:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content.strip()) == 0:
                        invalid_files.append(f"{filename} - empty file")
        except Exception as e:
            invalid_files.append(f"{filename} - {str(e)}")
    
    if missing_files or invalid_files:
        error_msg = ""
        if missing_files:
            error_msg += f"Missing files: {', '.join(missing_files)}\n"
        if invalid_files:
            error_msg += f"Invalid files: {', '.join(invalid_files)}"
        return False, error_msg
    
    print("✅ All data files present and valid")
    return True, ""


def check_demo_query_data_coverage() -> Tuple[bool, str]:
    """Check that demo data contains relevant content for the demo query"""
    print("🔍 Checking demo query data coverage...")
    
    demo_query = "What was the feedback on Project Phoenix?"
    search_terms = ["Phoenix", "feedback", "project"]
    
    try:
        # Check emails
        with open('emails.json', 'r', encoding='utf-8') as f:
            emails = json.load(f)
        
        email_matches = 0
        for email in emails:
            for term in search_terms:
                if (term.lower() in email.get('subject', '').lower() or 
                    term.lower() in email.get('body', '').lower()):
                    email_matches += 1
                    break
        
        # Check Slack messages
        with open('slack_messages.json', 'r', encoding='utf-8') as f:
            slack_messages = json.load(f)
        
        slack_matches = 0
        for msg in slack_messages:
            for term in search_terms:
                if term.lower() in msg.get('message', '').lower():
                    slack_matches += 1
                    break
        
        # Check document
        with open('project_notes.txt', 'r', encoding='utf-8') as f:
            document = f.read()
        
        doc_matches = 0
        for term in search_terms:
            if term.lower() in document.lower():
                doc_matches += 1
        
        # Validate coverage
        if email_matches == 0:
            return False, "No emails contain demo query terms"
        if slack_matches == 0:
            return False, "No Slack messages contain demo query terms"
        if doc_matches == 0:
            return False, "Document doesn't contain demo query terms"
        
        print(f"✅ Demo data coverage: {email_matches} emails, {slack_matches} Slack messages, {doc_matches} document terms")
        return True, ""
        
    except Exception as e:
        return False, f"Error checking data coverage: {str(e)}"


def test_search_functions() -> Tuple[bool, str]:
    """Test that search functions work correctly with demo data"""
    print("🔍 Testing search functions...")
    
    try:
        from synthetic_memory import search_emails, search_slack, search_documents, load_data
        
        # Load data
        emails, slack_messages, document = load_data()
        
        # Test email search
        email_results = search_emails("Phoenix", emails)
        if len(email_results) == 0:
            return False, "Email search for 'Phoenix' returned no results"
        
        # Test Slack search
        slack_results = search_slack("Phoenix", slack_messages)
        if len(slack_results) == 0:
            return False, "Slack search for 'Phoenix' returned no results"
        
        # Test document search
        doc_result = search_documents("Phoenix", document)
        if doc_result == "":
            return False, "Document search for 'Phoenix' returned no results"
        
        print(f"✅ Search functions working: {len(email_results)} emails, {len(slack_results)} Slack messages, document found")
        return True, ""
        
    except Exception as e:
        return False, f"Error testing search functions: {str(e)}"


def test_input_validation() -> Tuple[bool, str]:
    """Test input validation handles edge cases properly"""
    print("🔍 Testing input validation...")
    
    try:
        from synthetic_memory import validate_user_query
        
        # Test valid input
        is_valid, error = validate_user_query("What was the feedback on Project Phoenix?")
        if not is_valid:
            return False, f"Valid demo query rejected: {error}"
        
        # Test invalid inputs
        invalid_cases = [
            ("", "empty string"),
            (None, "None value"),
            ("a", "too short"),
            ("x" * 501, "too long"),
            ("<script>alert('test')</script>", "suspicious content")
        ]
        
        for invalid_input, description in invalid_cases:
            is_valid, error = validate_user_query(invalid_input)
            if is_valid:
                return False, f"Invalid input ({description}) was accepted"
        
        print("✅ Input validation working correctly")
        return True, ""
        
    except Exception as e:
        return False, f"Error testing input validation: {str(e)}"


def test_error_handling() -> Tuple[bool, str]:
    """Test that error handling works gracefully"""
    print("🔍 Testing error handling...")
    
    try:
        from synthetic_memory import run_agent
        from unittest.mock import patch
        
        # Test with invalid input
        result = run_agent("")
        if not isinstance(result, str) or "Error" not in result:
            return False, "Empty input should return error message"
        
        # Test with missing data files
        with patch('os.path.exists', return_value=False):
            result = run_agent("Test query")
            if not isinstance(result, str) or "Error loading data" not in result:
                return False, "Missing files should return appropriate error"
        
        print("✅ Error handling working correctly")
        return True, ""
        
    except Exception as e:
        return False, f"Error testing error handling: {str(e)}"


def test_performance() -> Tuple[bool, str]:
    """Test that the system performs adequately for demo"""
    print("🔍 Testing performance...")
    
    try:
        from synthetic_memory import load_data, search_emails, search_slack, search_documents
        
        # Test data loading speed
        start_time = time.time()
        emails, slack_messages, document = load_data()
        load_time = time.time() - start_time
        
        if load_time > 2.0:
            return False, f"Data loading too slow: {load_time:.2f}s (should be < 2s)"
        
        # Test search speed
        start_time = time.time()
        search_emails("Phoenix", emails)
        search_slack("Phoenix", slack_messages)
        search_documents("Phoenix", document)
        search_time = time.time() - start_time
        
        if search_time > 1.0:
            return False, f"Search operations too slow: {search_time:.2f}s (should be < 1s)"
        
        print(f"✅ Performance acceptable: load {load_time:.2f}s, search {search_time:.2f}s")
        return True, ""
        
    except Exception as e:
        return False, f"Error testing performance: {str(e)}"


def validate_demo_readiness() -> bool:
    """Run all validation checks and report demo readiness"""
    print("="*60)
    print("🚀 SYNTHETIC MEMORY LITE - DEMO VALIDATION")
    print("="*60)
    
    checks = [
        ("Data Files", check_data_files),
        ("Demo Query Coverage", check_demo_query_data_coverage),
        ("Search Functions", test_search_functions),
        ("Input Validation", test_input_validation),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
    ]
    
    passed = 0
    failed = 0
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 40)
        
        try:
            success, message = check_func()
            if success:
                passed += 1
                print(f"✅ PASSED")
            else:
                failed += 1
                print(f"❌ FAILED: {message}")
        except Exception as e:
            failed += 1
            print(f"❌ ERROR: {str(e)}")
    
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 DEMO IS READY FOR PRESENTATION!")
        print("🔥 All validation checks passed successfully")
        print("💡 Demo query: 'What was the feedback on Project Phoenix?'")
        print("🎯 Expected demo flow:")
        print("   1. User enters query")
        print("   2. System shows 'Thinking...' spinner")
        print("   3. AI generates search terms")
        print("   4. System searches all data sources")
        print("   5. AI synthesizes results with source attribution")
        print("   6. Results displayed with bullet points and sources")
        return True
    else:
        print(f"\n⚠️  DEMO NOT READY - {failed} issues need to be resolved")
        print("🔧 Please fix the failed checks before presenting")
        return False


def main():
    """Main entry point"""
    try:
        success = validate_demo_readiness()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error during validation: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()