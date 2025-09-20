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
        
        # Basic API key format validation (Google API keys typically start with 'AIza')
        api_key_stripped = api_key.strip()
        if len(api_key_stripped) < 20:
            raise ValueError("GEMINI_API_KEY appears to be too short to be valid")
        
        # Configure the Gemini API with timeout and retry settings
        genai.configure(api_key=api_key_stripped)
        
        # Initialize the GenerativeModel with 'gemini-pro'
        try:
            model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini model: {str(e)}")
        
        # Test the configuration with a minimal call (with timeout handling)
        try:
            import time
            start_time = time.time()
            
            # Make a minimal test call to verify the API is working
            test_response = model.generate_content("Test", 
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=10,
                    temperature=0.1
                ))
            
            elapsed_time = time.time() - start_time
            
            if not test_response:
                raise Exception("API test call returned empty response")
            
            if not test_response.text:
                raise Exception("API test call returned response without text content")
            
            # Log successful connection (for debugging)
            if elapsed_time > 5:
                st.warning(f"API response was slow ({elapsed_time:.1f}s). This may affect performance.")
                
        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise Exception(f"API authentication failed: Invalid API key or insufficient permissions")
            elif "quota" in error_msg or "limit" in error_msg:
                raise Exception(f"API quota exceeded: {str(e)}")
            elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                raise Exception(f"Network connectivity issue: {str(e)}")
            else:
                raise Exception(f"API configuration test failed: {str(e)}")
        
        return model
        
    except ValueError as e:
        st.error(f"🔑 API Key Error: {e}")
        st.info("**Configuration Help:**")
        st.write("1. Create a `.streamlit/secrets.toml` file in your project")
        st.write("2. Add your Gemini API key: `GEMINI_API_KEY = \"your-api-key-here\"`")
        st.write("3. Get an API key from: https://makersuite.google.com/app/apikey")
        raise
    except Exception as e:
        st.error(f"🤖 Gemini API Configuration Error: {e}")
        
        # Provide specific troubleshooting based on error type
        error_msg = str(e).lower()
        if "authentication" in error_msg or "api key" in error_msg:
            st.info("**API Key Issue:** Please verify your Gemini API key is correct and active.")
        elif "quota" in error_msg or "limit" in error_msg:
            st.info("**Quota Issue:** You may have exceeded your API usage limits. Check your Google Cloud console.")
        elif "network" in error_msg or "connection" in error_msg:
            st.info("**Network Issue:** Please check your internet connection and try again.")
        else:
            st.info("**General Issue:** Please check your API key and internet connection.")
        
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
    
    # Create the prompt for search term generation with better constraints
    prompt = f"""Your role is to analyze a user's query and determine the best keywords to search for in a database of emails, chat messages, and documents.

User Query: "{user_query}"

Please generate a simple JSON object containing a list of the most relevant and specific search strings. Focus on key terms, names, and concepts. Limit to 5 terms maximum. Only respond with the JSON, nothing else.

Format: {{ "search_terms": ["term1", "term2", "term3"] }}"""

    try:
        # Make API call to Gemini with generation config for better reliability
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.3,
                top_p=0.8
            )
        )
        
        if not response:
            raise Exception("No response received from Gemini API")
        
        if not response.text:
            raise Exception("Empty response text from Gemini API")
        
        response_text = response.text.strip()
        
        # Clean up response text (remove markdown code blocks if present)
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
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
                    clean_term = term.strip()
                    # Skip very short terms or common words
                    if len(clean_term) > 1 and clean_term.lower() not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                        valid_terms.append(clean_term)
            
            # Limit to maximum 5 terms
            return valid_terms[:5]
            
        except json.JSONDecodeError as e:
            # Enhanced fallback extraction
            st.warning(f"JSON parsing failed: {e}. Attempting intelligent fallback extraction.")
            
            # Try to extract terms from response text using multiple strategies
            fallback_terms = []
            
            # Strategy 1: Look for quoted terms
            import re
            quoted_terms = re.findall(r'"([^"]+)"', response_text)
            for term in quoted_terms:
                if len(term) > 1 and term not in fallback_terms:
                    fallback_terms.append(term)
            
            # Strategy 2: Look for terms in brackets
            bracket_terms = re.findall(r'\[([^\]]+)\]', response_text)
            for term in bracket_terms:
                clean_term = term.replace('"', '').replace("'", '').strip()
                if len(clean_term) > 1 and clean_term not in fallback_terms:
                    fallback_terms.append(clean_term)
            
            # Strategy 3: Extract meaningful words from response
            if not fallback_terms:
                words = response_text.replace('"', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').split()
                for word in words:
                    clean_word = word.strip('.,!?:;')
                    if len(clean_word) > 2 and clean_word.lower() not in ['search', 'terms', 'query', 'user', 'the', 'and', 'for']:
                        fallback_terms.append(clean_word)
            
            if fallback_terms:
                return fallback_terms[:5]  # Limit to 5 terms
            else:
                # Ultimate fallback: extract key words from original query
                words = user_query.split()
                return [word.strip('.,!?') for word in words if len(word) > 2][:3]
        
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "limit" in error_msg:
            st.error(f"API quota exceeded while generating search terms: {e}")
        elif "network" in error_msg or "connection" in error_msg:
            st.error(f"Network error while generating search terms: {e}")
        else:
            st.error(f"Error generating search terms: {e}")
        
        # Enhanced fallback: extract key words from original query
        words = user_query.split()
        fallback_terms = [word.strip('.,!?') for word in words if len(word) > 2][:3]
        
        if fallback_terms:
            st.info(f"Using fallback search terms extracted from your query: {', '.join(fallback_terms)}")
            return fallback_terms
        else:
            # Last resort: use the entire query as a single search term
            return [user_query.strip()]


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


def validate_user_query(user_query: str) -> tuple[bool, str]:
    """
    Validate user input query for safety and usability
    
    Args:
        user_query: The user's input query
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not user_query:
        return False, "Please provide a query to search for."
    
    if not isinstance(user_query, str):
        return False, "Query must be text."
    
    # Strip whitespace and check if empty
    query_stripped = user_query.strip()
    if not query_stripped:
        return False, "Please provide a non-empty query."
    
    # Check minimum length
    if len(query_stripped) < 2:
        return False, "Query is too short. Please provide at least 2 characters."
    
    # Check maximum length to prevent abuse
    if len(query_stripped) > 500:
        return False, "Query is too long. Please limit to 500 characters or less."
    
    # Check for potentially problematic characters or patterns
    suspicious_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
    query_lower = query_stripped.lower()
    for pattern in suspicious_patterns:
        if pattern in query_lower:
            return False, "Query contains potentially unsafe content. Please rephrase your question."
    
    # Check if query is just special characters or numbers
    if query_stripped.replace(' ', '').replace('.', '').replace('?', '').replace('!', '').isdigit():
        return False, "Please provide a descriptive question rather than just numbers."
    
    # Check for excessive repetition (same character repeated many times)
    for char in query_stripped:
        if query_stripped.count(char) > len(query_stripped) * 0.7:  # More than 70% same character
            return False, "Query appears to contain excessive repetition. Please provide a clear question."
    
    return True, ""


def run_agent(user_query: str) -> str:
    """
    Execute the complete agentic workflow: search term generation → multi-source search → result synthesis
    
    Args:
        user_query: The user's natural language query
        
    Returns:
        Final synthesized response with source attribution
        
    Requirements: 1.3, 1.4, 1.5, 1.6 - Complete agent workflow with error handling
    """
    # Enhanced input validation
    is_valid, validation_error = validate_user_query(user_query)
    if not is_valid:
        return f"❌ Input Error: {validation_error}"
    
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
    """Main Streamlit application entry point with comprehensive error handling"""
    try:
        # Task 6.1: Create main UI components
        st.title("Synthetic Memory Lite")
        st.markdown("*AI-powered information retrieval from your personal data*")
        
        # Add system status check at startup
        startup_errors = []
        
        # Check data files availability
        required_files = ["emails.json", "slack_messages.json", "project_notes.txt"]
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            startup_errors.append(f"Missing data files: {', '.join(missing_files)}")
        
        # Check API key availability
        if "GEMINI_API_KEY" not in st.secrets or not st.secrets["GEMINI_API_KEY"].strip():
            startup_errors.append("GEMINI_API_KEY not configured in Streamlit secrets")
        
        # Display startup errors if any
        if startup_errors:
            st.error("⚠️ **System Configuration Issues:**")
            for error in startup_errors:
                st.write(f"• {error}")
            st.write("Please resolve these issues before using the application.")
            st.stop()
        
        # Create text input field for user queries with pre-populated demo query
        user_query = st.text_input(
            "Enter your context or question:", 
            value="What was the feedback on Project Phoenix?",
            help="Ask questions about your emails, Slack messages, and documents. Keep queries clear and specific.",
            max_chars=500
        )
        
        # Show character count for user awareness
        if user_query:
            char_count = len(user_query)
            if char_count > 400:
                st.warning(f"Query length: {char_count}/500 characters")
            elif char_count > 0:
                st.caption(f"Query length: {char_count}/500 characters")
        
        # Add "Find Context" button to trigger agent workflow
        if st.button("Find Context", type="primary"):
            # Enhanced input validation
            is_valid, validation_error = validate_user_query(user_query)
            
            if not is_valid:
                st.error(f"❌ {validation_error}")
                return
            
            # Task 6.2: Implement user interaction flow
            # Add button click handler to execute run_agent() function
            # Implement st.spinner() with "Thinking..." message during processing
            with st.spinner("🤖 Analyzing your query and searching through your data..."):
                try:
                    # Execute the agent workflow
                    result = run_agent(user_query)
                    
                    # Display agent results using st.write() with proper formatting
                    st.success("✅ Search completed!")
                    st.write("### Results")
                    st.write(result)
                    
                except Exception as e:
                    # Add error display for failed operations
                    st.error(f"❌ An error occurred while processing your query: {str(e)}")
                    st.write("**Troubleshooting suggestions:**")
                    st.write("• Check your internet connection")
                    st.write("• Verify your Gemini API key is valid")
                    st.write("• Try rephrasing your query")
                    st.write("• Check that all data files are present and properly formatted")
                    
                    # Offer to show debug information
                    if st.button("Show Debug Information"):
                        st.write("**Error Details:**")
                        st.code(str(e))
        
        # Add helpful information section
        with st.expander("ℹ️ How to use Synthetic Memory Lite", expanded=False):
            st.write("""
            **What it does:**
            - Searches through your emails, Slack messages, and documents
            - Uses AI to understand your questions and find relevant information
            - Provides answers with clear source attribution
            
            **Example queries:**
            - "What was the feedback on Project Phoenix?"
            - "Who mentioned the budget concerns?"
            - "What are the key points from the project notes?"
            - "Any messages about the deadline?"
            
            **Tips for better results:**
            - Be specific about what you're looking for
            - Use key terms that might appear in your data
            - Ask about specific people, projects, or topics
            """)
        
    except Exception as e:
        st.error(f"❌ **Application Error:** {str(e)}")
        st.write("The application encountered an unexpected error during startup.")
        st.write("Please check your configuration and try refreshing the page.")
        
        # Show debug info for developers
        if st.checkbox("Show technical details"):
            st.code(f"Error: {str(e)}\nType: {type(e).__name__}")
            import traceback
            st.code(traceback.format_exc())
    
        # Optional: Add expandable section for debugging/testing (can be removed in production)
        with st.expander("🔧 System Status & Debug Information", expanded=False):
            st.write("This section shows system status for troubleshooting purposes.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📁 Data Files Status:**")
                # Test data loading functionality
                try:
                    email_data, slack_data, document_text = load_data()
                    
                    st.success("✅ Data loading successful!")
                    st.write(f"📧 Emails: {len(email_data)} loaded")
                    st.write(f"💬 Slack messages: {len(slack_data)} loaded") 
                    st.write(f"📄 Document: {len(document_text):,} characters")
                    
                    # Show sample data structure
                    if st.checkbox("Show sample data structure"):
                        if email_data:
                            st.write("**Sample email structure:**")
                            sample_email = {k: str(v)[:50] + "..." if len(str(v)) > 50 else v 
                                          for k, v in email_data[0].items()}
                            st.json(sample_email)
                        
                        if slack_data:
                            st.write("**Sample Slack message structure:**")
                            sample_slack = {k: str(v)[:50] + "..." if len(str(v)) > 50 else v 
                                          for k, v in slack_data[0].items()}
                            st.json(sample_slack)
                    
                except Exception as e:
                    st.error(f"❌ Data loading failed: {e}")
                    st.write("Check that all required data files are present and properly formatted.")
            
            with col2:
                st.write("**🤖 API Configuration Status:**")
                # Test Gemini API configuration
                try:
                    model = configure_gemini_api()
                    st.success("✅ Gemini API connected!")
                    st.write("🔗 Model: gemini-pro")
                    
                    # Test API responsiveness
                    if st.button("Test API Response Time"):
                        import time
                        start_time = time.time()
                        try:
                            test_response = model.generate_content("Hello", 
                                generation_config=genai.types.GenerationConfig(max_output_tokens=5))
                            response_time = time.time() - start_time
                            st.write(f"⚡ Response time: {response_time:.2f}s")
                            if response_time > 3:
                                st.warning("API response is slow")
                            else:
                                st.success("API response is fast")
                        except Exception as e:
                            st.error(f"API test failed: {e}")
                    
                except Exception as e:
                    st.error(f"❌ Gemini API failed: {e}")
                    st.write("Check your API key configuration in Streamlit secrets.")
            
            # System information
            st.write("**💻 System Information:**")
            import sys
            import platform
            st.write(f"• Python version: {sys.version.split()[0]}")
            st.write(f"• Platform: {platform.system()} {platform.release()}")
            st.write(f"• Streamlit version: {st.__version__}")
            
            # Environment check
            st.write("**🔐 Environment Check:**")
            secrets_status = "✅ Configured" if "GEMINI_API_KEY" in st.secrets else "❌ Missing"
            st.write(f"• GEMINI_API_KEY: {secrets_status}")
            
            files_status = []
            for file in ["emails.json", "slack_messages.json", "project_notes.txt"]:
                status = "✅" if os.path.exists(file) else "❌"
                files_status.append(f"• {file}: {status}")
            
            for status in files_status:
                st.write(status)


if __name__ == "__main__":
    main()