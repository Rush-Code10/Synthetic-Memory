"""
Synthetic Memory Lite - AI-powered information retrieval and synthesis demo

This Streamlit application demonstrates agentic AI capabilities by intelligently
searching through emails, chat messages, and documents to provide contextual
answers to user queries using Google Gemini API.
"""

import streamlit as st
import json
import google.generativeai as genai
from typing import List, Dict, Tuple
import os


def load_data() -> Tuple[List[Dict], List[Dict], str]:
    """
    Load and parse the three data files: emails.json, slack_messages.json, and project_notes.txt
    
    Returns:
        tuple: (email_data, slack_data, document_text)
        - email_data: List of email dictionaries with 'from', 'date', 'subject', 'body' fields
        - slack_data: List of Slack message dictionaries with 'channel', 'user', 'date', 'message' fields  
        - document_text: String containing the full text of project_notes.txt
        
    Raises:
        FileNotFoundError: If any required data file is missing
        json.JSONDecodeError: If JSON files are corrupted or invalid
        Exception: For other file reading errors
    """
    email_data = []
    slack_data = []
    document_text = ""
    
    try:
        # Load emails.json
        if not os.path.exists("emails.json"):
            raise FileNotFoundError("emails.json file not found")
            
        with open("emails.json", "r", encoding="utf-8") as f:
            email_data = json.load(f)
            
        # Validate email data structure
        if not isinstance(email_data, list):
            raise ValueError("emails.json must contain a list of email objects")
            
        for i, email in enumerate(email_data):
            if not isinstance(email, dict):
                raise ValueError(f"Email at index {i} is not a dictionary")
            required_fields = ["from", "date", "subject", "body"]
            for field in required_fields:
                if field not in email:
                    raise ValueError(f"Email at index {i} missing required field: {field}")
                    
    except FileNotFoundError as e:
        st.error(f"Email data file error: {e}")
        raise
    except json.JSONDecodeError as e:
        st.error(f"Error parsing emails.json: {e}")
        raise
    except ValueError as e:
        st.error(f"Email data validation error: {e}")
        raise
    except Exception as e:
        st.error(f"Unexpected error loading email data: {e}")
        raise
        
    try:
        # Load slack_messages.json
        if not os.path.exists("slack_messages.json"):
            raise FileNotFoundError("slack_messages.json file not found")
            
        with open("slack_messages.json", "r", encoding="utf-8") as f:
            slack_data = json.load(f)
            
        # Validate slack data structure
        if not isinstance(slack_data, list):
            raise ValueError("slack_messages.json must contain a list of message objects")
            
        for i, message in enumerate(slack_data):
            if not isinstance(message, dict):
                raise ValueError(f"Slack message at index {i} is not a dictionary")
            required_fields = ["channel", "user", "date", "message"]
            for field in required_fields:
                if field not in message:
                    raise ValueError(f"Slack message at index {i} missing required field: {field}")
                    
    except FileNotFoundError as e:
        st.error(f"Slack data file error: {e}")
        raise
    except json.JSONDecodeError as e:
        st.error(f"Error parsing slack_messages.json: {e}")
        raise
    except ValueError as e:
        st.error(f"Slack data validation error: {e}")
        raise
    except Exception as e:
        st.error(f"Unexpected error loading Slack data: {e}")
        raise
        
    try:
        # Load project_notes.txt
        if not os.path.exists("project_notes.txt"):
            raise FileNotFoundError("project_notes.txt file not found")
            
        with open("project_notes.txt", "r", encoding="utf-8") as f:
            document_text = f.read()
            
        # Validate document text
        if not isinstance(document_text, str):
            raise ValueError("project_notes.txt must contain text data")
            
        if len(document_text.strip()) == 0:
            raise ValueError("project_notes.txt appears to be empty")
            
    except FileNotFoundError as e:
        st.error(f"Document file error: {e}")
        raise
    except Exception as e:
        st.error(f"Unexpected error loading document: {e}")
        raise
    
    # Final validation - ensure we have data
    if len(email_data) == 0:
        st.warning("No email data loaded")
    if len(slack_data) == 0:
        st.warning("No Slack message data loaded")
    if len(document_text.strip()) == 0:
        st.warning("No document text loaded")
        
    return email_data, slack_data, document_text


def configure_gemini_api():
    """
    Configure the Gemini API client using the API key from Streamlit secrets
    
    Returns:
        genai.GenerativeModel: Configured Gemini model instance
        
    Raises:
        ValueError: If API key is missing from secrets
        Exception: For API configuration errors
    """
    try:
        # Check if API key exists in secrets
        if "GEMINI_API_KEY" not in st.secrets:
            raise ValueError("GEMINI_API_KEY not found in Streamlit secrets. Please add it to your secrets.toml file.")
        
        api_key = st.secrets["GEMINI_API_KEY"]
        
        # Validate API key is not empty
        if not api_key or api_key.strip() == "":
            raise ValueError("GEMINI_API_KEY is empty in Streamlit secrets")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the GenerativeModel with 'gemini-pro'
        model = genai.GenerativeModel('gemini-pro')
        
        # Test the configuration with a simple call
        try:
            # Make a minimal test call to verify the API is working
            test_response = model.generate_content("Hello")
            if not test_response:
                raise Exception("API test call returned empty response")
        except Exception as e:
            raise Exception(f"API configuration test failed: {str(e)}")
        
        return model
        
    except ValueError as e:
        st.error(f"API Key Error: {e}")
        st.info("Please ensure GEMINI_API_KEY is properly set in your Streamlit secrets configuration.")
        raise
    except Exception as e:
        st.error(f"Gemini API Configuration Error: {e}")
        st.info("Please check your API key and internet connection.")
        raise


def search_emails(query: str, email_data: List[Dict]) -> List[Dict]:
    """
    Search email data for query matches using case-insensitive substring matching
    
    Args:
        query: Search term to look for
        email_data: List of email dictionaries with 'from', 'date', 'subject', 'body' fields
        
    Returns:
        List of matching email dictionaries with original structure preserved
        
    Requirements: 2.2 - Search both "subject" and "body" fields for query matches
    """
    if not query or not isinstance(query, str):
        return []
    
    if not email_data or not isinstance(email_data, list):
        return []
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower().strip()
    
    if not query_lower:
        return []
    
    matching_emails = []
    
    for email in email_data:
        if not isinstance(email, dict):
            continue
            
        # Check if query matches in subject field
        subject_match = False
        if 'subject' in email and isinstance(email['subject'], str):
            if query_lower in email['subject'].lower():
                subject_match = True
        
        # Check if query matches in body field
        body_match = False
        if 'body' in email and isinstance(email['body'], str):
            if query_lower in email['body'].lower():
                body_match = True
        
        # Add email to results if match found in either subject or body
        if subject_match or body_match:
            matching_emails.append(email)
    
    return matching_emails


def search_slack(query: str, slack_data: List[Dict]) -> List[Dict]:
    """
    Search Slack message data for query matches using case-insensitive substring matching
    
    Args:
        query: Search term to look for
        slack_data: List of Slack message dictionaries with 'channel', 'user', 'date', 'message' fields
        
    Returns:
        List of matching Slack message dictionaries with original structure preserved
        
    Requirements: 2.3 - Search "message" field for query matches
    """
    if not query or not isinstance(query, str):
        return []
    
    if not slack_data or not isinstance(slack_data, list):
        return []
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower().strip()
    
    if not query_lower:
        return []
    
    matching_messages = []
    
    for message in slack_data:
        if not isinstance(message, dict):
            continue
            
        # Check if query matches in message field
        if 'message' in message and isinstance(message['message'], str):
            if query_lower in message['message'].lower():
                matching_messages.append(message)
    
    return matching_messages


def search_documents(query: str, document_text: str) -> str:
    """
    Search document text for query matches using case-insensitive substring matching
    
    Args:
        query: Search term to look for
        document_text: Full text content of the document
        
    Returns:
        Full document text if match found, empty string if not
        
    Requirements: 2.4 - Search entire document text for query matches
    """
    if not query or not isinstance(query, str):
        return ""
    
    if not document_text or not isinstance(document_text, str):
        return ""
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower().strip()
    
    if not query_lower:
        return ""
    
    # Check if query matches anywhere in the document text
    if query_lower in document_text.lower():
        return document_text
    else:
        return ""


def main():
    """Main Streamlit application entry point"""
    st.title("Synthetic Memory Lite")
    
    # Test data loading functionality
    try:
        email_data, slack_data, document_text = load_data()
        
        st.success("✅ Data loading successful!")
        st.write(f"📧 Loaded {len(email_data)} emails")
        st.write(f"💬 Loaded {len(slack_data)} Slack messages") 
        st.write(f"📄 Loaded document with {len(document_text)} characters")
        
    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
        st.write("Please check that all required data files are present and properly formatted.")
        return
    
    # Test Gemini API configuration
    try:
        model = configure_gemini_api()
        st.success("✅ Gemini API configuration successful!")
        st.write("🤖 Connected to gemini-pro model")
        
    except Exception as e:
        st.error(f"❌ Gemini API configuration failed: {e}")
        st.write("Please check your API key configuration in Streamlit secrets.")
        return
    
    # Show sample data for verification
    if st.checkbox("Show sample data"):
        st.subheader("Sample Email")
        if email_data:
            st.json(email_data[0])
            
        st.subheader("Sample Slack Message")
        if slack_data:
            st.json(slack_data[0])
            
        st.subheader("Document Preview")
        st.text(document_text[:500] + "..." if len(document_text) > 500 else document_text)
    
    # Test search functionality
    if st.checkbox("Test search functions"):
        st.subheader("Search Function Testing")
        
        test_query = st.text_input("Enter test search query:", value="Phoenix")
        
        if test_query:
            # Test email search
            email_results = search_emails(test_query, email_data)
            st.write(f"📧 Email search results: {len(email_results)} matches")
            if email_results:
                for i, email in enumerate(email_results[:2]):  # Show first 2 results
                    st.write(f"**Email {i+1}:** {email['subject']}")
            
            # Test Slack search
            slack_results = search_slack(test_query, slack_data)
            st.write(f"💬 Slack search results: {len(slack_results)} matches")
            if slack_results:
                for i, msg in enumerate(slack_results[:2]):  # Show first 2 results
                    st.write(f"**Message {i+1}:** {msg['message'][:100]}...")
            
            # Test document search
            doc_result = search_documents(test_query, document_text)
            st.write(f"📄 Document search: {'Match found' if doc_result else 'No match'}")
            if doc_result:
                st.write("Document contains the search term")


if __name__ == "__main__":
    main()