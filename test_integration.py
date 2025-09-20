"""
Integration tests for Synthetic Memory Lite - Demo validation and end-to-end testing

This test suite covers:
- Complete workflow with pre-populated demo query
- Source attribution verification in synthesized responses  
- Error handling scenarios and edge cases
- UI responsiveness and loading indicators validation
- Demo presentation readiness

Requirements: 4.4, 5.2, 5.3, 5.4, 5.5
"""

import unittest
import json
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict

# Import the functions we want to test
from synthetic_memory import (
    run_agent,
    load_data,
    configure_gemini_api,
    generate_search_terms,
    synthesize_results,
    search_emails,
    search_slack,
    search_documents,
    validate_user_query
)


class TestCompleteWorkflow(unittest.TestCase):
    """Test complete workflow with pre-populated demo query"""
    
    def setUp(self):
        """Set up test environment with sample data"""
        # Demo query from requirements
        self.demo_query = "What was the feedback on Project Phoenix?"
        
        # Sample data matching the actual data files
        self.sample_emails = [
            {
                "from": "sarah.chen@company.com",
                "date": "2024-03-12",
                "subject": "Project Phoenix - Initial Feedback",
                "body": "Hi team, I've reviewed the Project Phoenix proposal and I'm impressed with the technical approach. The architecture looks solid and the timeline seems realistic."
            },
            {
                "from": "alex.kim@company.com",
                "date": "2024-03-18",
                "subject": "Project Phoenix - Technical Deep Dive",
                "body": "Team, I've completed the technical review of Project Phoenix. The proposed microservices architecture is well-designed, but I have concerns about the database scaling approach."
            }
        ]
        
        self.sample_slack = [
            {
                "channel": "#project-phoenix",
                "user": "sarah_chen",
                "date": "2024-03-13",
                "message": "Just finished reviewing the Phoenix proposal. Really solid work! 👏 My only concern is we might be rushing the user testing phase."
            },
            {
                "channel": "#project-phoenix",
                "user": "alex_kim",
                "date": "2024-03-18",
                "message": "Technical review is done. Overall positive but we need to address the database scaling strategy."
            }
        ]
        
        self.sample_document = "PROJECT PHOENIX - INTERNAL DOCUMENTATION\n\nProject Phoenix is a next-generation customer engagement platform. Technical Review (Alex Kim): Architecture approved, database scaling concerns noted. UX Review (Sarah Chen): Strong foundation, needs more user testing phases."
    
    @patch('synthetic_memory.configure_gemini_api')
    @patch('synthetic_memory.load_data')
    def test_demo_query_complete_workflow(self, mock_load_data, mock_configure_api):
        """Test complete workflow with the pre-populated demo query"""
        # Mock data loading
        mock_load_data.return_value = (self.sample_emails, self.sample_slack, self.sample_document)
        
        # Mock Gemini API configuration
        mock_model = Mock()
        mock_configure_api.return_value = mock_model
        
        # Mock search term generation
        mock_search_response = Mock()
        mock_search_response.text = '{"search_terms": ["Phoenix", "feedback", "project"]}'
        
        # Mock synthesis response
        mock_synthesis_response = Mock()
        mock_synthesis_response.text = """Based on the data provided:
• Project Phoenix received positive feedback on technical approach and architecture
• Concerns raised about database scaling and user testing phases
• Overall assessment is positive with specific areas for improvement

Sources:
• Email from sarah.chen@company.com: March 12 - Initial feedback on technical approach
• Email from alex.kim@company.com: March 18 - Technical review with database concerns
• Slack message from sarah_chen in #project-phoenix: March 13 - Positive review with user testing concerns
• Document: Project Phoenix internal documentation"""
        
        # Configure mock to return different responses for different calls
        mock_model.generate_content.side_effect = [mock_search_response, mock_synthesis_response]
        
        # Execute the complete workflow
        result = run_agent(self.demo_query)
        
        # Verify the workflow executed successfully
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)  # Should have substantial content
        
        # Verify data loading was called
        mock_load_data.assert_called_once()
        
        # Verify API configuration was called
        mock_configure_api.assert_called_once()
        
        # Verify Gemini was called twice (search terms + synthesis)
        self.assertEqual(mock_model.generate_content.call_count, 2)
        
        # Verify result contains expected content
        self.assertIn("Phoenix", result)
        self.assertIn("feedback", result)
        
        print(f"✅ Demo query workflow test passed. Result length: {len(result)} characters")
    
    @patch('synthetic_memory.configure_gemini_api')
    @patch('synthetic_memory.load_data')
    def test_workflow_with_search_results(self, mock_load_data, mock_configure_api):
        """Test workflow produces results that match search expectations"""
        # Mock data loading
        mock_load_data.return_value = (self.sample_emails, self.sample_slack, self.sample_document)
        
        # Mock Gemini API
        mock_model = Mock()
        mock_configure_api.return_value = mock_model
        
        # Mock responses
        mock_search_response = Mock()
        mock_search_response.text = '{"search_terms": ["Phoenix", "feedback"]}'
        
        mock_synthesis_response = Mock()
        mock_synthesis_response.text = "Synthesized response with source attribution"
        
        mock_model.generate_content.side_effect = [mock_search_response, mock_synthesis_response]
        
        # Execute workflow
        result = run_agent(self.demo_query)
        
        # Verify search terms were used effectively
        # The search should find relevant emails and Slack messages
        self.assertIsInstance(result, str)
        self.assertNotIn("Error", result)  # Should not contain error messages
        
        print("✅ Workflow search results test passed")
    
    @patch('synthetic_memory.configure_gemini_api')
    @patch('synthetic_memory.load_data')
    def test_workflow_performance_timing(self, mock_load_data, mock_configure_api):
        """Test workflow completes in reasonable time for demo"""
        # Mock data loading
        mock_load_data.return_value = (self.sample_emails, self.sample_slack, self.sample_document)
        
        # Mock Gemini API with realistic delays
        mock_model = Mock()
        mock_configure_api.return_value = mock_model
        
        def mock_generate_with_delay(*args, **kwargs):
            time.sleep(0.1)  # Simulate API delay
            mock_response = Mock()
            if "search_terms" in str(args):
                mock_response.text = '{"search_terms": ["Phoenix", "feedback"]}'
            else:
                mock_response.text = "Quick synthesized response for demo"
            return mock_response
        
        mock_model.generate_content.side_effect = mock_generate_with_delay
        
        # Measure execution time
        start_time = time.time()
        result = run_agent(self.demo_query)
        execution_time = time.time() - start_time
        
        # Verify reasonable execution time (should be under 5 seconds for demo)
        self.assertLess(execution_time, 5.0, f"Workflow took {execution_time:.2f}s, too slow for demo")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        
        print(f"✅ Workflow performance test passed. Execution time: {execution_time:.2f}s")


class TestSourceAttribution(unittest.TestCase):
    """Test proper source attribution in synthesized responses"""
    
    def setUp(self):
        """Set up test data for source attribution testing"""
        self.test_emails = [
            {
                "from": "sarah.chen@company.com",
                "date": "2024-03-12",
                "subject": "Project Phoenix Feedback",
                "body": "The technical approach is solid"
            }
        ]
        
        self.test_slack = [
            {
                "channel": "#project-phoenix",
                "user": "alex_kim",
                "date": "2024-03-18",
                "message": "Database concerns need addressing"
            }
        ]
        
        self.test_document = "Project Phoenix documentation with technical details"
    
    def test_source_attribution_format(self):
        """Test that source attribution follows the expected format"""
        mock_model = Mock()
        
        # Mock synthesis response with proper source attribution
        mock_response = Mock()
        mock_response.text = """Based on the data:
• Technical approach is solid
• Database concerns identified

Sources:
• Email from sarah.chen@company.com: March 12
• Slack message from alex_kim in #project-phoenix: March 18
• Document: Project Phoenix documentation"""
        
        mock_model.generate_content.return_value = mock_response
        
        result = synthesize_results(
            "What was the feedback?",
            self.test_emails,
            self.test_slack,
            self.test_document,
            mock_model
        )
        
        # Verify source attribution format
        self.assertIn("Sources:", result)
        self.assertIn("Email from sarah.chen@company.com", result)
        self.assertIn("Slack message from alex_kim", result)
        self.assertIn("Document:", result)
        
        print("✅ Source attribution format test passed")
    
    def test_source_attribution_accuracy(self):
        """Test that source attribution matches actual data sources"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Response with sources"
        mock_model.generate_content.return_value = mock_response
        
        # Test with specific data
        result = synthesize_results(
            "Test query",
            self.test_emails,
            self.test_slack,
            self.test_document,
            mock_model
        )
        
        # Verify the synthesis call included correct source data
        call_args = mock_model.generate_content.call_args[0][0]
        
        # Check that email data is properly formatted in the prompt
        self.assertIn("sarah.chen@company.com", call_args)
        self.assertIn("2024-03-12", call_args)
        self.assertIn("Project Phoenix Feedback", call_args)
        
        # Check that Slack data is properly formatted
        self.assertIn("#project-phoenix", call_args)
        self.assertIn("alex_kim", call_args)
        self.assertIn("Database concerns", call_args)
        
        # Check that document data is included
        self.assertIn("Project Phoenix documentation", call_args)
        
        print("✅ Source attribution accuracy test passed")
    
    def test_empty_sources_handling(self):
        """Test source attribution when some sources are empty"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Response with limited sources"
        mock_model.generate_content.return_value = mock_response
        
        # Test with empty Slack and document
        result = synthesize_results(
            "Test query",
            self.test_emails,
            [],  # Empty Slack
            "",  # Empty document
            mock_model
        )
        
        # Verify the call handled empty sources properly
        call_args = mock_model.generate_content.call_args[0][0]
        self.assertIn("No relevant Slack messages found", call_args)
        self.assertIn("No relevant document content found", call_args)
        
        print("✅ Empty sources handling test passed")


class TestErrorHandlingScenarios(unittest.TestCase):
    """Test error handling scenarios and edge cases"""
    
    def test_missing_data_files_error_handling(self):
        """Test graceful handling of missing data files"""
        with patch('os.path.exists', return_value=False):
            result = run_agent("Test query")
            
            # Should return error message, not crash
            self.assertIsInstance(result, str)
            self.assertIn("Error loading data", result)
            self.assertIn("ensure all data files are present", result)
        
        print("✅ Missing data files error handling test passed")
    
    def test_api_configuration_error_handling(self):
        """Test graceful handling of API configuration errors"""
        with patch('synthetic_memory.load_data') as mock_load:
            mock_load.return_value = ([], [], "")
            
            with patch('synthetic_memory.configure_gemini_api') as mock_config:
                mock_config.side_effect = Exception("API key invalid")
                
                result = run_agent("Test query")
                
                # Should return error message, not crash
                self.assertIsInstance(result, str)
                self.assertIn("Error configuring Gemini API", result)
                self.assertIn("API key", result)
        
        print("✅ API configuration error handling test passed")
    
    def test_search_term_generation_failure(self):
        """Test handling of search term generation failures"""
        with patch('synthetic_memory.load_data') as mock_load:
            mock_load.return_value = ([], [], "")
            
            with patch('synthetic_memory.configure_gemini_api') as mock_config:
                mock_model = Mock()
                mock_config.return_value = mock_model
                
                # Mock search term generation failure
                mock_model.generate_content.side_effect = Exception("API quota exceeded")
                
                result = run_agent("Test query")
                
                # Should handle gracefully and use fallback
                self.assertIsInstance(result, str)
                # Should not crash, should attempt fallback processing
                self.assertNotIn("Traceback", result)
        
        print("✅ Search term generation failure handling test passed")
    
    def test_synthesis_failure_fallback(self):
        """Test fallback behavior when synthesis fails"""
        with patch('synthetic_memory.load_data') as mock_load:
            mock_load.return_value = ([{"from": "test", "date": "2024", "subject": "test", "body": "test"}], [], "")
            
            with patch('synthetic_memory.configure_gemini_api') as mock_config:
                mock_model = Mock()
                mock_config.return_value = mock_model
                
                # Mock successful search term generation but failed synthesis
                mock_search_response = Mock()
                mock_search_response.text = '{"search_terms": ["test"]}'
                
                mock_model.generate_content.side_effect = [
                    mock_search_response,  # Successful search terms
                    Exception("Synthesis failed")  # Failed synthesis
                ]
                
                result = run_agent("Test query")
                
                # Should provide fallback response
                self.assertIsInstance(result, str)
                self.assertIn("Query:", result)  # Fallback format
                self.assertIn("Emails Found", result)
        
        print("✅ Synthesis failure fallback test passed")
    
    def test_invalid_user_input_handling(self):
        """Test handling of various invalid user inputs"""
        invalid_inputs = [
            "",  # Empty string
            None,  # None value
            "a",  # Too short
            "x" * 501,  # Too long
            "<script>alert('test')</script>",  # Suspicious content
        ]
        
        for invalid_input in invalid_inputs:
            result = run_agent(invalid_input)
            
            # Should return appropriate error message
            self.assertIsInstance(result, str)
            self.assertIn("Error", result)
            # Should not crash or attempt processing
            self.assertNotIn("Traceback", result)
        
        print("✅ Invalid user input handling test passed")
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted data files"""
        with patch('synthetic_memory.load_data') as mock_load:
            mock_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            
            result = run_agent("Test query")
            
            # Should handle gracefully
            self.assertIsInstance(result, str)
            self.assertIn("Error loading data", result)
        
        print("✅ Corrupted data handling test passed")


class TestUIResponsivenessValidation(unittest.TestCase):
    """Test UI responsiveness and loading indicators validation"""
    
    def test_loading_state_simulation(self):
        """Test that loading states would be properly handled"""
        # This test simulates the Streamlit loading behavior
        # In actual Streamlit, this would test st.spinner() functionality
        
        with patch('synthetic_memory.load_data') as mock_load:
            with patch('synthetic_memory.configure_gemini_api') as mock_config:
                # Simulate slow operations
                def slow_load():
                    time.sleep(0.1)  # Simulate loading delay
                    return ([], [], "")
                
                def slow_config():
                    time.sleep(0.1)  # Simulate API config delay
                    mock_model = Mock()
                    mock_response = Mock()
                    mock_response.text = '{"search_terms": ["test"]}'
                    mock_model.generate_content.return_value = mock_response
                    return mock_model
                
                mock_load.side_effect = slow_load
                mock_config.side_effect = slow_config
                
                # Measure response time
                start_time = time.time()
                result = run_agent("Test query")
                response_time = time.time() - start_time
                
                # Verify reasonable response time
                self.assertLess(response_time, 2.0, "Response too slow for good UX")
                self.assertIsInstance(result, str)
        
        print("✅ Loading state simulation test passed")
    
    def test_progress_indication_readiness(self):
        """Test that the system is ready for progress indication"""
        # This test verifies that operations are structured to support progress indication
        
        # Test that each major step can be isolated for progress tracking
        with patch('synthetic_memory.load_data') as mock_load:
            mock_load.return_value = ([], [], "")
            
            # Step 1: Data loading (should be quick)
            start = time.time()
            try:
                load_data()
            except:
                pass  # Expected to fail in test environment
            load_time = time.time() - start
            self.assertLess(load_time, 0.5, "Data loading too slow")
            
            # Step 2: API configuration (should be quick when mocked)
            with patch('synthetic_memory.configure_gemini_api') as mock_config:
                mock_config.return_value = Mock()
                start = time.time()
                try:
                    configure_gemini_api()
                except:
                    pass  # Expected to fail without proper secrets in test
                config_time = time.time() - start
                self.assertLess(config_time, 0.5, "API config too slow")
        
        print("✅ Progress indication readiness test passed")


class TestDemoPresentationReadiness(unittest.TestCase):
    """Test demo presentation readiness and smooth execution"""
    
    def setUp(self):
        """Set up realistic demo environment"""
        self.demo_query = "What was the feedback on Project Phoenix?"
        
        # Use actual data structure from the real files
        self.demo_emails = [
            {
                "from": "sarah.chen@company.com",
                "date": "2024-03-12",
                "subject": "Project Phoenix - Initial Feedback",
                "body": "Hi team, I've reviewed the Project Phoenix proposal and I'm impressed with the technical approach."
            }
        ]
        
        self.demo_slack = [
            {
                "channel": "#project-phoenix",
                "user": "sarah_chen",
                "date": "2024-03-13",
                "message": "Just finished reviewing the Phoenix proposal. Really solid work!"
            }
        ]
        
        self.demo_document = "PROJECT PHOENIX - INTERNAL DOCUMENTATION\n\nFeedback Summary: Technical Review (Alex Kim): Architecture approved"
    
    @patch('synthetic_memory.configure_gemini_api')
    @patch('synthetic_memory.load_data')
    def test_demo_query_produces_meaningful_results(self, mock_load_data, mock_configure_api):
        """Test that demo query produces meaningful, presentable results"""
        # Mock data loading with demo data
        mock_load_data.return_value = (self.demo_emails, self.demo_slack, self.demo_document)
        
        # Mock Gemini API
        mock_model = Mock()
        mock_configure_api.return_value = mock_model
        
        # Mock realistic demo responses
        mock_search_response = Mock()
        mock_search_response.text = '{"search_terms": ["Phoenix", "feedback", "project"]}'
        
        mock_synthesis_response = Mock()
        mock_synthesis_response.text = """Based on the data provided:

• Project Phoenix received positive feedback on the technical approach
• The architecture was praised as solid and well-designed
• Team members expressed enthusiasm about the project's potential

Sources:
• Email from sarah.chen@company.com: March 12 - Initial feedback praising technical approach
• Slack message from sarah_chen in #project-phoenix: March 13 - Positive review of proposal
• Document: Project Phoenix internal documentation with feedback summary"""
        
        mock_model.generate_content.side_effect = [mock_search_response, mock_synthesis_response]
        
        # Execute demo query
        result = run_agent(self.demo_query)
        
        # Verify demo-quality results
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100, "Result too short for meaningful demo")
        
        # Verify key demo elements are present
        self.assertIn("Phoenix", result)
        self.assertIn("feedback", result)
        self.assertIn("Sources:", result)
        self.assertIn("sarah.chen@company.com", result)
        
        # Verify professional presentation format
        self.assertIn("•", result)  # Bullet points for readability
        
        print("✅ Demo query meaningful results test passed")
        print(f"Demo result preview: {result[:200]}...")
    
    def test_demo_data_completeness(self):
        """Test that demo data provides comprehensive search results"""
        # Test email search with demo query terms
        email_results = search_emails("Phoenix", self.demo_emails)
        self.assertGreater(len(email_results), 0, "Demo emails should contain Phoenix references")
        
        email_results = search_emails("feedback", self.demo_emails)
        self.assertGreater(len(email_results), 0, "Demo emails should contain feedback references")
        
        # Test Slack search
        slack_results = search_slack("Phoenix", self.demo_slack)
        self.assertGreater(len(slack_results), 0, "Demo Slack should contain Phoenix references")
        
        # Test document search
        doc_result = search_documents("Phoenix", self.demo_document)
        self.assertNotEqual(doc_result, "", "Demo document should contain Phoenix references")
        
        print("✅ Demo data completeness test passed")
    
    @patch('synthetic_memory.configure_gemini_api')
    @patch('synthetic_memory.load_data')
    def test_demo_consistency_multiple_runs(self, mock_load_data, mock_configure_api):
        """Test that demo produces consistent results across multiple runs"""
        # Mock consistent setup
        mock_load_data.return_value = (self.demo_emails, self.demo_slack, self.demo_document)
        
        mock_model = Mock()
        mock_configure_api.return_value = mock_model
        
        # Mock consistent responses with longer content
        mock_search_response = Mock()
        mock_search_response.text = '{"search_terms": ["Phoenix", "feedback"]}'
        
        mock_synthesis_response = Mock()
        mock_synthesis_response.text = """Consistent demo response with comprehensive source attribution.

Based on the data provided:
• Project Phoenix received positive feedback on technical approach
• Architecture was praised as solid and well-designed
• Team expressed enthusiasm about project potential

Sources:
• Email from sarah.chen@company.com: March 12 - Technical feedback
• Slack message from sarah_chen in #project-phoenix: March 13"""
        
        # Set up mock to return responses for multiple calls
        mock_model.generate_content.side_effect = [
            mock_search_response, mock_synthesis_response,  # Run 1
            mock_search_response, mock_synthesis_response,  # Run 2
            mock_search_response, mock_synthesis_response   # Run 3
        ]
        
        # Run demo query multiple times
        results = []
        for i in range(3):
            result = run_agent(self.demo_query)
            results.append(result)
        
        # Verify all runs succeeded
        for i, result in enumerate(results):
            self.assertIsInstance(result, str, f"Run {i+1} failed")
            self.assertGreater(len(result), 50, f"Run {i+1} result too short: {len(result)} chars")
            self.assertNotIn("Error", result, f"Run {i+1} contained errors")
        
        print("✅ Demo consistency test passed")
    
    def test_demo_error_recovery(self):
        """Test that demo can recover from common errors gracefully"""
        # Test with various error scenarios that might occur during demo
        
        # Scenario 1: Temporary API issue
        with patch('synthetic_memory.load_data') as mock_load:
            mock_load.return_value = (self.demo_emails, self.demo_slack, self.demo_document)
            
            with patch('synthetic_memory.configure_gemini_api') as mock_config:
                mock_config.side_effect = Exception("Temporary API issue")
                
                result = run_agent(self.demo_query)
                
                # Should provide graceful error message suitable for demo
                self.assertIsInstance(result, str)
                self.assertIn("Error configuring Gemini API", result)
                self.assertNotIn("Traceback", result)  # No technical stack traces
        
        print("✅ Demo error recovery test passed")


def run_integration_tests():
    """Run all integration tests and provide summary"""
    print("="*60)
    print("RUNNING INTEGRATION TESTS FOR SYNTHETIC MEMORY LITE")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCompleteWorkflow,
        TestSourceAttribution,
        TestErrorHandlingScenarios,
        TestUIResponsivenessValidation,
        TestDemoPresentationReadiness
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nIntegration Tests: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)