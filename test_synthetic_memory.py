"""
Unit tests for Synthetic Memory Lite core functionality

This test suite covers:
- Data loading functions with sample data
- Search functions with known queries and expected results  
- JSON parsing for Gemini API responses
- Mocked Gemini API calls for consistent testing

Requirements: Testing Strategy from Design Document
"""

import unittest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

# Import the functions we want to test
from synthetic_memory import (
    load_data,
    search_emails,
    search_slack,
    search_documents,
    generate_search_terms,
    synthesize_results,
    validate_user_query
)


class TestDataLoading(unittest.TestCase):
    """Test data loading functions with sample data"""
    
    def setUp(self):
        """Set up test data files"""
        # Sample email data
        self.sample_emails = [
            {
                "from": "test@example.com",
                "date": "2024-03-01",
                "subject": "Test Subject",
                "body": "Test email body content"
            },
            {
                "from": "user@company.com", 
                "date": "2024-03-02",
                "subject": "Project Update",
                "body": "Project is progressing well"
            }
        ]
        
        # Sample Slack data
        self.sample_slack = [
            {
                "channel": "#general",
                "user": "testuser",
                "date": "2024-03-01",
                "message": "Hello team, how is everyone doing?"
            },
            {
                "channel": "#project",
                "user": "developer",
                "date": "2024-03-02", 
                "message": "The new feature is ready for testing"
            }
        ]
        
        # Sample document text
        self.sample_document = "This is a test document with project information and important details."
        
        # Create temporary files
        self.temp_dir = tempfile.mkdtemp()
        self.emails_file = os.path.join(self.temp_dir, "emails.json")
        self.slack_file = os.path.join(self.temp_dir, "slack_messages.json")
        self.doc_file = os.path.join(self.temp_dir, "project_notes.txt")
        
        # Write test data to files
        with open(self.emails_file, 'w') as f:
            json.dump(self.sample_emails, f)
        
        with open(self.slack_file, 'w') as f:
            json.dump(self.sample_slack, f)
            
        with open(self.doc_file, 'w') as f:
            f.write(self.sample_document)
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_data_success(self, mock_json_load, mock_open, mock_exists):
        """Test successful data loading"""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock file reading
        mock_json_load.side_effect = [self.sample_emails, self.sample_slack]
        mock_open.return_value.__enter__.return_value.read.return_value = self.sample_document
        
        # Call function
        emails, slack, doc = load_data()
        
        # Verify results
        self.assertEqual(emails, self.sample_emails)
        self.assertEqual(slack, self.sample_slack)
        self.assertEqual(doc, self.sample_document)
    
    @patch('os.path.exists')
    def test_load_data_missing_file(self, mock_exists):
        """Test handling of missing data files"""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            load_data()
    
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_data_invalid_json(self, mock_json_load, mock_open, mock_exists):
        """Test handling of corrupted JSON files"""
        mock_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        with self.assertRaises(json.JSONDecodeError):
            load_data()
    
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_data_invalid_structure(self, mock_json_load, mock_open, mock_exists):
        """Test handling of invalid data structure"""
        mock_exists.return_value = True
        # Return invalid structure (not a list)
        mock_json_load.side_effect = [{"invalid": "structure"}, self.sample_slack]
        
        with self.assertRaises(ValueError):
            load_data()
    
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_data_missing_required_fields(self, mock_json_load, mock_open, mock_exists):
        """Test handling of data with missing required fields"""
        mock_exists.return_value = True
        # Email missing required field
        invalid_emails = [{"from": "test@example.com", "date": "2024-03-01"}]  # Missing subject and body
        mock_json_load.side_effect = [invalid_emails, self.sample_slack]
        
        with self.assertRaises(ValueError):
            load_data()


class TestSearchFunctions(unittest.TestCase):
    """Test search functions with known queries and expected results"""
    
    def setUp(self):
        """Set up test data"""
        self.test_emails = [
            {
                "from": "alice@company.com",
                "date": "2024-03-01",
                "subject": "Project Phoenix Update",
                "body": "The Phoenix project is making great progress. We've completed the initial phase."
            },
            {
                "from": "bob@company.com",
                "date": "2024-03-02", 
                "subject": "Budget Review",
                "body": "Need to review the quarterly budget allocations for all projects."
            },
            {
                "from": "carol@company.com",
                "date": "2024-03-03",
                "subject": "Team Meeting",
                "body": "Let's schedule a team meeting to discuss Phoenix project milestones."
            }
        ]
        
        self.test_slack = [
            {
                "channel": "#general",
                "user": "alice",
                "date": "2024-03-01",
                "message": "Great work on the Phoenix project everyone!"
            },
            {
                "channel": "#budget",
                "user": "bob", 
                "date": "2024-03-02",
                "message": "The budget review meeting is scheduled for Friday."
            },
            {
                "channel": "#phoenix",
                "user": "carol",
                "date": "2024-03-03",
                "message": "Phoenix milestone 1 completed successfully."
            }
        ]
        
        self.test_document = "Project Phoenix Documentation\n\nThis document contains important information about the Phoenix project timeline, budget, and technical specifications. The project aims to deliver innovative solutions."
    
    def test_search_emails_subject_match(self):
        """Test email search with subject field matches"""
        results = search_emails("Phoenix", self.test_emails)
        
        # Should find 2 emails with "Phoenix" in subject or body
        self.assertEqual(len(results), 2)
        self.assertIn("Phoenix", results[0]["subject"])
        self.assertIn("Phoenix", results[1]["body"])
    
    def test_search_emails_body_match(self):
        """Test email search with body field matches"""
        results = search_emails("budget", self.test_emails)
        
        # Should find 1 email with "budget" in body
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["from"], "bob@company.com")
    
    def test_search_emails_case_insensitive(self):
        """Test email search is case insensitive"""
        results_upper = search_emails("PHOENIX", self.test_emails)
        results_lower = search_emails("phoenix", self.test_emails)
        results_mixed = search_emails("Phoenix", self.test_emails)
        
        # All should return same results
        self.assertEqual(len(results_upper), 2)
        self.assertEqual(len(results_lower), 2)
        self.assertEqual(len(results_mixed), 2)
        self.assertEqual(results_upper, results_lower)
        self.assertEqual(results_lower, results_mixed)
    
    def test_search_emails_no_match(self):
        """Test email search with no matches"""
        results = search_emails("nonexistent", self.test_emails)
        self.assertEqual(len(results), 0)
    
    def test_search_emails_empty_query(self):
        """Test email search with empty query"""
        results = search_emails("", self.test_emails)
        self.assertEqual(len(results), 0)
        
        results = search_emails(None, self.test_emails)
        self.assertEqual(len(results), 0)
    
    def test_search_emails_invalid_data(self):
        """Test email search with invalid data"""
        results = search_emails("test", None)
        self.assertEqual(len(results), 0)
        
        results = search_emails("test", [])
        self.assertEqual(len(results), 0)
        
        # Test with malformed email data
        bad_data = [{"invalid": "structure"}, "not a dict"]
        results = search_emails("test", bad_data)
        self.assertEqual(len(results), 0)
    
    def test_search_slack_message_match(self):
        """Test Slack search with message matches"""
        results = search_slack("Phoenix", self.test_slack)
        
        # Should find 2 messages with "Phoenix"
        self.assertEqual(len(results), 2)
        self.assertIn("Phoenix", results[0]["message"])
        self.assertIn("Phoenix", results[1]["message"])
    
    def test_search_slack_case_insensitive(self):
        """Test Slack search is case insensitive"""
        results_upper = search_slack("BUDGET", self.test_slack)
        results_lower = search_slack("budget", self.test_slack)
        
        self.assertEqual(len(results_upper), 1)
        self.assertEqual(len(results_lower), 1)
        self.assertEqual(results_upper, results_lower)
    
    def test_search_slack_no_match(self):
        """Test Slack search with no matches"""
        results = search_slack("nonexistent", self.test_slack)
        self.assertEqual(len(results), 0)
    
    def test_search_slack_empty_query(self):
        """Test Slack search with empty query"""
        results = search_slack("", self.test_slack)
        self.assertEqual(len(results), 0)
        
        results = search_slack(None, self.test_slack)
        self.assertEqual(len(results), 0)
    
    def test_search_documents_match(self):
        """Test document search with matches"""
        result = search_documents("Phoenix", self.test_document)
        self.assertEqual(result, self.test_document)
        
        result = search_documents("timeline", self.test_document)
        self.assertEqual(result, self.test_document)
    
    def test_search_documents_case_insensitive(self):
        """Test document search is case insensitive"""
        result_upper = search_documents("PHOENIX", self.test_document)
        result_lower = search_documents("phoenix", self.test_document)
        
        self.assertEqual(result_upper, self.test_document)
        self.assertEqual(result_lower, self.test_document)
        self.assertEqual(result_upper, result_lower)
    
    def test_search_documents_no_match(self):
        """Test document search with no matches"""
        result = search_documents("nonexistent", self.test_document)
        self.assertEqual(result, "")
    
    def test_search_documents_empty_query(self):
        """Test document search with empty query"""
        result = search_documents("", self.test_document)
        self.assertEqual(result, "")
        
        result = search_documents(None, self.test_document)
        self.assertEqual(result, "")
    
    def test_search_documents_invalid_data(self):
        """Test document search with invalid data"""
        result = search_documents("test", None)
        self.assertEqual(result, "")
        
        result = search_documents("test", "")
        self.assertEqual(result, "")


class TestGeminiAPIIntegration(unittest.TestCase):
    """Test JSON parsing for Gemini API responses and mocked API calls"""
    
    def setUp(self):
        """Set up mock Gemini model"""
        self.mock_model = Mock()
        self.test_query = "What was the feedback on Project Phoenix?"
    
    def test_generate_search_terms_valid_json(self):
        """Test search term generation with valid JSON response"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.text = '{"search_terms": ["Phoenix", "feedback", "project"]}'
        self.mock_model.generate_content.return_value = mock_response
        
        result = generate_search_terms(self.test_query, self.mock_model)
        
        self.assertEqual(result, ["Phoenix", "feedback", "project"])
        self.mock_model.generate_content.assert_called_once()
    
    def test_generate_search_terms_json_with_markdown(self):
        """Test search term generation with JSON wrapped in markdown"""
        # Mock API response with markdown code blocks
        mock_response = Mock()
        mock_response.text = '```json\n{"search_terms": ["Phoenix", "feedback"]}\n```'
        self.mock_model.generate_content.return_value = mock_response
        
        result = generate_search_terms(self.test_query, self.mock_model)
        
        self.assertEqual(result, ["Phoenix", "feedback"])
    
    def test_generate_search_terms_invalid_json(self):
        """Test search term generation with invalid JSON response"""
        # Mock API response with invalid JSON
        mock_response = Mock()
        mock_response.text = 'Invalid JSON response with terms: Phoenix, feedback, project'
        self.mock_model.generate_content.return_value = mock_response
        
        result = generate_search_terms(self.test_query, self.mock_model)
        
        # Should fall back to extracting terms from response text
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    def test_generate_search_terms_empty_response(self):
        """Test search term generation with empty response"""
        # Mock empty API response
        mock_response = Mock()
        mock_response.text = ""
        self.mock_model.generate_content.return_value = mock_response
        
        result = generate_search_terms(self.test_query, self.mock_model)
        
        # Should fall back to extracting from original query
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)  # Should have some terms from fallback
    
    def test_generate_search_terms_no_response(self):
        """Test search term generation with no response"""
        # Mock no response
        self.mock_model.generate_content.return_value = None
        
        result = generate_search_terms(self.test_query, self.mock_model)
        
        # Should fall back to extracting from original query
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    def test_generate_search_terms_api_error(self):
        """Test search term generation with API error"""
        # Mock API error
        self.mock_model.generate_content.side_effect = Exception("API Error")
        
        result = generate_search_terms(self.test_query, self.mock_model)
        
        # Should fall back to extracting from original query
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    def test_generate_search_terms_malformed_json_structure(self):
        """Test search term generation with malformed JSON structure"""
        # Mock response with wrong JSON structure
        mock_response = Mock()
        mock_response.text = '{"wrong_field": ["Phoenix", "feedback"]}'
        self.mock_model.generate_content.return_value = mock_response
        
        result = generate_search_terms(self.test_query, self.mock_model)
        
        # Should fall back to extracting from response text
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    def test_generate_search_terms_empty_query(self):
        """Test search term generation with empty query"""
        result = generate_search_terms("", self.mock_model)
        self.assertEqual(result, [])
        
        result = generate_search_terms(None, self.mock_model)
        self.assertEqual(result, [])
    
    def test_generate_search_terms_no_model(self):
        """Test search term generation with no model"""
        with self.assertRaises(ValueError):
            generate_search_terms(self.test_query, None)
    
    def test_synthesize_results_success(self):
        """Test result synthesis with successful API response"""
        # Mock successful synthesis response
        mock_response = Mock()
        mock_response.text = "Based on the data provided:\n• Phoenix project received positive feedback\n• Technical approach was praised\n\nSources:\n• Email from sarah.chen@company.com: March 12"
        self.mock_model.generate_content.return_value = mock_response
        
        # Test data
        email_results = [{"from": "sarah.chen@company.com", "date": "March 12", "subject": "Phoenix Feedback", "body": "Great work on Phoenix"}]
        slack_results = []
        document_result = ""
        
        result = synthesize_results(self.test_query, email_results, slack_results, document_result, self.mock_model)
        
        self.assertIn("Phoenix project", result)
        self.assertIn("sarah.chen@company.com", result)
        self.mock_model.generate_content.assert_called_once()
    
    def test_synthesize_results_api_error(self):
        """Test result synthesis with API error"""
        # Mock API error
        self.mock_model.generate_content.side_effect = Exception("API Error")
        
        email_results = [{"from": "test@example.com", "date": "2024-03-01", "subject": "Test", "body": "Test content"}]
        slack_results = []
        document_result = ""
        
        result = synthesize_results(self.test_query, email_results, slack_results, document_result, self.mock_model)
        
        # Should return fallback response
        self.assertIn("Query:", result)
        self.assertIn("Emails Found", result)
    
    def test_synthesize_results_empty_response(self):
        """Test result synthesis with empty API response"""
        # Mock empty response
        mock_response = Mock()
        mock_response.text = ""
        self.mock_model.generate_content.return_value = mock_response
        
        email_results = []
        slack_results = []
        document_result = ""
        
        result = synthesize_results(self.test_query, email_results, slack_results, document_result, self.mock_model)
        
        # Should return fallback response
        self.assertIn("No relevant information found", result)
    
    def test_synthesize_results_no_model(self):
        """Test result synthesis with no model"""
        with self.assertRaises(ValueError):
            synthesize_results(self.test_query, [], [], "", None)


class TestInputValidation(unittest.TestCase):
    """Test user input validation"""
    
    def test_validate_user_query_valid(self):
        """Test validation of valid queries"""
        valid_queries = [
            "What was the feedback on Project Phoenix?",
            "Show me budget information",
            "Who worked on the technical review?",
            "Timeline for the project"
        ]
        
        for query in valid_queries:
            is_valid, error = validate_user_query(query)
            self.assertTrue(is_valid, f"Query '{query}' should be valid")
            self.assertEqual(error, "")
    
    def test_validate_user_query_empty(self):
        """Test validation of empty queries"""
        invalid_queries = [None, "", "   ", "\t\n"]
        
        for query in invalid_queries:
            is_valid, error = validate_user_query(query)
            self.assertFalse(is_valid)
            self.assertIn("query", error.lower())  # Error message contains "query"
    
    def test_validate_user_query_too_short(self):
        """Test validation of too short queries"""
        is_valid, error = validate_user_query("a")
        self.assertFalse(is_valid)
        self.assertIn("too short", error.lower())
    
    def test_validate_user_query_too_long(self):
        """Test validation of too long queries"""
        long_query = "a" * 501  # Over 500 character limit
        is_valid, error = validate_user_query(long_query)
        self.assertFalse(is_valid)
        self.assertIn("too long", error.lower())
    
    def test_validate_user_query_suspicious_content(self):
        """Test validation of potentially unsafe queries"""
        suspicious_queries = [
            "<script>alert('test')</script>",
            "javascript:alert('test')",
            "eval(malicious_code)",
            "exec(dangerous_command)"
        ]
        
        for query in suspicious_queries:
            is_valid, error = validate_user_query(query)
            self.assertFalse(is_valid)
            self.assertIn("unsafe", error.lower())
    
    def test_validate_user_query_only_numbers(self):
        """Test validation of queries with only numbers"""
        is_valid, error = validate_user_query("12345")
        self.assertFalse(is_valid)
        self.assertIn("descriptive", error.lower())
    
    def test_validate_user_query_excessive_repetition(self):
        """Test validation of queries with excessive character repetition"""
        is_valid, error = validate_user_query("aaaaaaaaaaaaaaaaaaa")
        self.assertFalse(is_valid)
        self.assertIn("repetition", error.lower())


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)