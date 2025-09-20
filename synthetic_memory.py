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


def generate_search_terms(user_query: str, model: genai.GenerativeModel) -> List[str]:
    """
    Use Gemini to generate relevant search terms from a user query
    
    Args:
        user_query: The user's natural language query
        model: Configured Gemini GenerativeModel instance
        
    Returns:
        List of search terms extracted from the query
        
    Requirements: 2.1, 5.1 - Generate appropriate search terms using Gemini AI
    """
    if not user_query or not isinstance(user_query, str):
        return []
    
    if not model:
        raise ValueError("Gemini model not provided")
    
    # Create the prompt for search term generation
    prompt = f"""Your role is to analyze a user's query and determine the best keywords to search for in a database of emails, chat messages, and documents.

User Query: "{user_query}"

Please generate a simple JSON object containing a list of the most relevant and specific search strings. Only respond with the JSON, nothing else.

Format: {{ "search_terms": ["term1", "term2", "term3"] }}"""

    try:
        # Make API call to Gemini
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini API")
        
        response_text = response.text.strip()
        
        # Parse JSON response
        try:
            parsed_response = json.loads(response_text)
            
            # Validate response structure
            if not isinstance(parsed_response, dict):
                raise ValueError("Response is not a JSON object")
            
            if "search_terms" not in parsed_response:
                raise ValueError("Response missing 'search_terms' field")
            
            search_terms = parsed_response["search_terms"]
            
            if not isinstance(search_terms, list):
                raise ValueError("'search_terms' field is not a list")
            
            # Filter and validate search terms
            valid_terms = []
            for term in search_terms:
                if isinstance(term, str) and term.strip():
                    valid_terms.append(term.strip())
            
            return valid_terms
            
        except json.JSONDecodeError as e:
            # Try to extract search terms from malformed JSON
            st.warning(f"JSON parsing failed: {e}. Attempting fallback extraction.")
            
            # Fallback: try to extract terms from response text
            fallback_terms = []
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('{') and not line.startswith('}'):
                    # Remove quotes and common JSON artifacts
                    clean_line = line.replace('"', '').replace(',', '').replace('[', '').replace(']', '')
                    if clean_line and len(clean_line) > 1:
                        fallback_terms.append(clean_line)
            
            if fallback_terms:
                return fallback_terms[:5]  # Limit to 5 terms
            else:
                # Ultimate fallback: extract key words from original query
                words = user_query.split()
                return [word.strip('.,!?') for word in words if len(word) > 2][:3]
        
    except Exception as e:
        st.error(f"Error generating search terms: {e}")
        # Fallback: extract key words from original query
        words = user_query.split()
        fallback_terms = [word.strip('.,!?') for word in words if len(word) > 2][:3]
        st.warning(f"Using fallback search terms: {fallback_terms}")
        return fallback_terms


def synthesize_results(user_query: str, email_results: List[Dict], slack_results: List[Dict], 
                      document_result: str, model: genai.GenerativeModel) -> str:
    """
    Use Gemini to synthesize search results into a coherent response with source attribution
    
    Args:
        user_query: The original user query
        email_results: List of matching email dictionaries
        slack_results: List of matching Slack message dictionaries  
        document_result: Document text if match found, empty string if not
        model: Configured Gemini GenerativeModel instance
        
    Returns:
        Synthesized response with proper source attribution
        
    Requirements: 2.5, 2.6, 5.6 - Synthesize results with source attribution, no hallucination
    """
    if not model:
        raise ValueError("Gemini model not provided")
    
    # Prepare data for synthesis
    email_data_str = ""
    if email_results:
        email_summaries = []
        for email in email_results:
            summary = f"From: {email.get('from', 'Unknown')}, Date: {email.get('date', 'Unknown')}, Subject: {email.get('subject', 'No subject')}, Body: {email.get('body', 'No content')[:200]}..."
            email_summaries.append(summary)
        email_data_str = "\n".join(email_summaries)
    else:
        email_data_str = "No relevant emails found."
    
    slack_data_str = ""
    if slack_results:
        slack_summaries = []
        for msg in slack_results:
            summary = f"Channel: {msg.get('channel', 'Unknown')}, User: {msg.get('user', 'Unknown')}, Date: {msg.get('date', 'Unknown')}, Message: {msg.get('message', 'No content')[:200]}..."
            slack_summaries.append(summary)
        slack_data_str = "\n".join(slack_summaries)
    else:
        slack_data_str = "No relevant Slack messages found."
    
    document_data_str = ""
    if document_result:
        # Truncate document for synthesis if too long
        if len(document_result) > 1000:
            document_data_str = document_result[:1000] + "..."
        else:
            document_data_str = document_result
    else:
        document_data_str = "No relevant document content found."
    
    # Create synthesis prompt
    prompt = f"""Act as a helpful AI assistant. Synthesize the following information from the user's personal data to directly answer their query.

User's Query: "{user_query}"

Data to synthesize:
- EMAILS: {email_data_str}
- SLACK MESSAGES: {slack_data_str}
- DOCUMENT: {document_data_str}

Provide a very concise, bullet-point summary answer based ONLY on the data provided. Then, clearly list the sources (e.g., "Email from Sarah Chen: March 12"). Do not make anything up."""

    try:
        # Make API call to Gemini for synthesis
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini API")
        
        synthesized_response = response.text.strip()
        
        # Validate that we got a meaningful response
        if len(synthesized_response) < 10:
            raise Exception("Response too short, likely incomplete")
        
        return synthesized_response
        
    except Exception as e:
        st.error(f"Error synthesizing results: {e}")
        
        # Fallback: create a basic summary without AI synthesis
        fallback_response = f"**Query:** {user_query}\n\n"
        
        if email_results:
            fallback_response += f"**Emails Found ({len(email_results)}):**\n"
            for i, email in enumerate(email_results[:3]):  # Show max 3
                fallback_response += f"• Email from {email.get('from', 'Unknown')} ({email.get('date', 'Unknown')}): {email.get('subject', 'No subject')}\n"
            if len(email_results) > 3:
                fallback_response += f"• ... and {len(email_results) - 3} more emails\n"
            fallback_response += "\n"
        
        if slack_results:
            fallback_response += f"**Slack Messages Found ({len(slack_results)}):**\n"
            for i, msg in enumerate(slack_results[:3]):  # Show max 3
                fallback_response += f"• Message from {msg.get('user', 'Unknown')} in #{msg.get('channel', 'Unknown')} ({msg.get('date', 'Unknown')})\n"
            if len(slack_results) > 3:
                fallback_response += f"• ... and {len(slack_results) - 3} more messages\n"
            fallback_response += "\n"
        
        if document_result:
            fallback_response += "**Document:** Relevant content found in project notes\n\n"
        
        if not email_results and not slack_results and not document_result:
            fallback_response += "**No relevant information found** in emails, Slack messages, or documents for this query.\n"
        
        st.warning("Using fallback response due to synthesis error.")
        return fallback_response


def run_agent(user_query: str) -> str:
    """
    Execute the complete agentic workflow: search term generation → multi-source search → result synthesis
    
    Args:
        user_query: The user's natural language query
        
    Returns:
        Final synthesized response with source attribution
        
    Requirements: 1.3, 1.4, 1.5, 1.6 - Complete agent workflow with error handling
    """
    if not user_query or not isinstance(user_query, str) or not user_query.strip():
        return "Please provide a valid query to search for."
    
    try:
        # Step 1: Load data
        try:
            email_data, slack_data, document_text = load_data()
        except Exception as e:
            return f"Error loading data: {e}. Please ensure all data files are present and properly formatted."
        
        # Step 2: Configure Gemini API
        try:
            model = configure_gemini_api()
        except Exception as e:
            return f"Error configuring Gemini API: {e}. Please check your API key configuration."
        
        # Step 3: Generate search terms using Gemini
        try:
            search_terms = generate_search_terms(user_query, model)
            if not search_terms:
                return "Unable to generate search terms from your query. Please try rephrasing your question."
        except Exception as e:
            st.error(f"Search term generation failed: {e}")
            # Fallback: use original query words as search terms
            search_terms = [word.strip('.,!?') for word in user_query.split() if len(word) > 2][:3]
            if not search_terms:
                return "Unable to process your query. Please try rephrasing your question."
        
        # Step 4: Execute multi-source search using generated terms
        all_email_results = []
        all_slack_results = []
        document_result = ""
        
        try:
            # Search across all sources with each search term
            for term in search_terms:
                # Search emails
                email_matches = search_emails(term, email_data)
                for email in email_matches:
                    if email not in all_email_results:  # Avoid duplicates
                        all_email_results.append(email)
                
                # Search Slack messages
                slack_matches = search_slack(term, slack_data)
                for msg in slack_matches:
                    if msg not in all_slack_results:  # Avoid duplicates
                        all_slack_results.append(msg)
                
                # Search documents (if any term matches, we get the full document)
                if not document_result:  # Only search if we haven't found a match yet
                    doc_match = search_documents(term, document_text)
                    if doc_match:
                        document_result = doc_match
                        
        except Exception as e:
            st.error(f"Search execution failed: {e}")
            return f"Error occurred while searching data sources: {e}"
        
        # Step 5: Synthesize results using Gemini
        try:
            synthesized_response = synthesize_results(
                user_query, 
                all_email_results, 
                all_slack_results, 
                document_result, 
                model
            )
            return synthesized_response
            
        except Exception as e:
            st.error(f"Result synthesis failed: {e}")
            
            # Final fallback: return raw search results
            fallback = f"**Search Results for:** {user_query}\n\n"
            fallback += f"**Search Terms Used:** {', '.join(search_terms)}\n\n"
            
            if all_email_results:
                fallback += f"**Found {len(all_email_results)} relevant emails:**\n"
                for email in all_email_results[:3]:
                    fallback += f"• {email.get('subject', 'No subject')} from {email.get('from', 'Unknown')}\n"
                if len(all_email_results) > 3:
                    fallback += f"• ... and {len(all_email_results) - 3} more\n"
                fallback += "\n"
            
            if all_slack_results:
                fallback += f"**Found {len(all_slack_results)} relevant Slack messages:**\n"
                for msg in all_slack_results[:3]:
                    fallback += f"• Message from {msg.get('user', 'Unknown')} in #{msg.get('channel', 'Unknown')}\n"
                if len(all_slack_results) > 3:
                    fallback += f"• ... and {len(all_slack_results) - 3} more\n"
                fallback += "\n"
            
            if document_result:
                fallback += "**Found relevant content in project documents**\n\n"
            
            if not all_email_results and not all_slack_results and not document_result:
                fallback += "**No relevant information found** for your query.\n"
            
            return fallback
            
    except Exception as e:
        st.error(f"Unexpected error in agent workflow: {e}")
        return f"An unexpected error occurred: {e}. Please try again or contact support."


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