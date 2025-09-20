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
from datetime import datetime


class AgentThought:
    """Represents a single step in the agent's thinking process"""
    
    def __init__(self, step_name: str, input_data: any, output_data: any, 
                 timestamp: datetime = None, metadata: Dict = None):
        self.step_name = step_name
        self.input_data = input_data
        self.output_data = output_data
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'step_name': self.step_name,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


def add_agent_thought(step_name: str, input_data: any, output_data: any, metadata: Dict = None):
    """Add a thought to the agent's thinking process in session state"""
    if 'agent_thoughts' not in st.session_state:
        st.session_state.agent_thoughts = []
    
    thought = AgentThought(step_name, input_data, output_data, metadata=metadata)
    st.session_state.agent_thoughts.append(thought)


def clear_agent_thoughts():
    """Clear all agent thoughts from session state"""
    if 'agent_thoughts' in st.session_state:
        st.session_state.agent_thoughts = []


def tag_search_results_with_ids(email_results: List[Dict], slack_results: List[Dict], document_result: str) -> Tuple[List[Dict], List[Dict], str, Dict]:
    """
    Add unique IDs to search results and create a source mapping
    
    Args:
        email_results: List of email dictionaries
        slack_results: List of Slack message dictionaries
        document_result: Document text if found
        
    Returns:
        Tuple of (tagged_emails, tagged_slack, tagged_document, source_mapping)
    """
    source_mapping = {}
    current_id = 1
    
    # Tag email results
    tagged_emails = []
    for email in email_results:
        tagged_email = email.copy()
        tagged_email['_source_id'] = current_id
        tagged_email['_source_type'] = 'email'
        source_mapping[current_id] = {
            'type': 'email',
            'data': tagged_email,
            'description': f"Email from {email.get('from', 'Unknown')} on {email.get('date', 'Unknown')}"
        }
        tagged_emails.append(tagged_email)
        current_id += 1
    
    # Tag Slack results
    tagged_slack = []
    for msg in slack_results:
        tagged_msg = msg.copy()
        tagged_msg['_source_id'] = current_id
        tagged_msg['_source_type'] = 'slack'
        source_mapping[current_id] = {
            'type': 'slack',
            'data': tagged_msg,
            'description': f"Slack message from {msg.get('user', 'Unknown')} in #{msg.get('channel', 'Unknown')} on {msg.get('date', 'Unknown')}"
        }
        tagged_slack.append(tagged_msg)
        current_id += 1
    
    # Tag document result
    tagged_document = document_result
    if document_result:
        source_mapping[current_id] = {
            'type': 'document',
            'data': {'content': document_result, 'name': 'project_notes.txt'},
            'description': "Project documentation and notes"
        }
        tagged_document = f"[DOCUMENT_ID:{current_id}] {document_result}"
        current_id += 1
    
    return tagged_emails, tagged_slack, tagged_document, source_mapping


def display_answer_with_sources(final_response: Dict, source_mapping: Dict):
    """
    Display the answer with inline citations and expandable source details
    
    Args:
        final_response: Dictionary with 'answer' and 'sources' keys
        source_mapping: Mapping of source IDs to source data
    """
    if not isinstance(final_response, dict) or 'answer' not in final_response:
        st.error("Invalid response format")
        return
    
    answer = final_response.get('answer', '')
    sources = final_response.get('sources', [])
    
    # Display the answer with superscript citations
    st.markdown("### Answer")
    
    # Convert [1] to superscript ¹ using Markdown
    import re
    answer_with_superscript = re.sub(r'\[(\d+)\]', r'^\1^', answer)
    st.markdown(answer_with_superscript)
    
    # Display sources section
    if sources:
        st.markdown("### Sources")
        
        for i, source_ref in enumerate(sources, 1):
            source_id = source_ref.get('id')
            source_type = source_ref.get('type', 'unknown')
            content_snippet = source_ref.get('content_snippet', '')
            
            if source_id in source_mapping:
                source_data = source_mapping[source_id]
                source_description = source_data['description']
                
                # Create expandable source section
                with st.expander(f"[{i}] {source_type.title()}: {source_description}", expanded=False):
                    st.markdown(f"**Content Snippet:** {content_snippet}")
                    st.markdown("**Full Source Data:**")
                    st.json(source_data['data'])
            else:
                # Fallback if source ID not found
                with st.expander(f"[{i}] {source_type.title()}: Source not found", expanded=False):
                    st.warning(f"Source ID {source_id} not found in mapping")
                    st.json(source_ref)


def show_agent_thinking():
    """Display the agent's thinking process in an expandable UI component"""
    if 'agent_thoughts' not in st.session_state or not st.session_state.agent_thoughts:
        return
    
    with st.expander("🧠 See the Agent's Thinking", expanded=False):
        st.markdown("**Here's how the AI agent processed your query:**")
        
        for i, thought in enumerate(st.session_state.agent_thoughts, 1):
            with st.container():
                # Step header
                st.markdown(f"### Step {i}: {thought.step_name}")
                
                # Input section
                st.markdown("**📥 Input:**")
                if isinstance(thought.input_data, str):
                    if len(thought.input_data) > 200:
                        st.text_area("", thought.input_data, height=100, key=f"input_{i}", disabled=True)
                    else:
                        st.write(thought.input_data)
                elif isinstance(thought.input_data, list):
                    if len(thought.input_data) > 10:
                        st.write(f"List with {len(thought.input_data)} items:")
                        st.json(thought.input_data[:5])
                        if len(thought.input_data) > 5:
                            st.write(f"... and {len(thought.input_data) - 5} more items")
                    else:
                        st.json(thought.input_data)
                else:
                    st.json(thought.input_data)
                
                # Output section
                st.markdown("**📤 Output:**")
                if isinstance(thought.output_data, str):
                    if len(thought.output_data) > 300:
                        st.text_area("", thought.output_data, height=150, key=f"output_{i}", disabled=True)
                    else:
                        st.write(thought.output_data)
                elif isinstance(thought.output_data, list):
                    if len(thought.output_data) > 10:
                        st.write(f"List with {len(thought.output_data)} items:")
                        st.json(thought.output_data[:5])
                        if len(thought.output_data) > 5:
                            st.write(f"... and {len(thought.output_data) - 5} more items")
                    else:
                        st.json(thought.output_data)
                else:
                    st.json(thought.output_data)
                
                # Metadata section (if any)
                if thought.metadata:
                    st.markdown("**ℹ️ Additional Info:**")
                    for key, value in thought.metadata.items():
                        st.write(f"• **{key}**: {value}")
                
                # Add separator between steps
                if i < len(st.session_state.agent_thoughts):
                    st.markdown("---")
        
        # Summary
        st.markdown("**📊 Summary:**")
        st.write(f"• Total steps: {len(st.session_state.agent_thoughts)}")
        st.write(f"• Processing time: {st.session_state.agent_thoughts[-1].timestamp - st.session_state.agent_thoughts[0].timestamp}")
        
        # Clear thoughts button
        if st.button("🗑️ Clear Thinking History", key="clear_thoughts"):
            clear_agent_thoughts()
            st.rerun()


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
        api_key = st.secrets["GEMINI_API_KEY"]
        
        # Check if API key exists in secrets
        if "GEMINI_API_KEY" not in st.secrets:
            raise ValueError("GEMINI_API_KEY not found in Streamlit secrets. Please add it to your secrets.toml file.")
        
        # Validate API key is not empty
        if not api_key or api_key.strip() == "":
            raise ValueError("GEMINI_API_KEY is empty in Streamlit secrets")
        
        # Basic API key format validation (Google API keys typically start with 'AIza')
        api_key_stripped = api_key.strip()
        if len(api_key_stripped) < 20:
            raise ValueError("GEMINI_API_KEY appears to be too short to be valid")
        
        # Configure the Gemini API with timeout and retry settings
        genai.configure(api_key=api_key_stripped)
        
        # Initialize the GenerativeModel with 'gemini-1.5-flash' (free tier)
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
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


def synthesize_results_with_citations(user_query: str, email_results: List[Dict], slack_results: List[Dict], 
                                    document_result: str, model: genai.GenerativeModel) -> Dict:
    """
    Use Gemini to synthesize search results into a JSON response with inline citations
    
    Args:
        user_query: The original user query
        email_results: List of matching email dictionaries with source IDs
        slack_results: List of matching Slack message dictionaries with source IDs
        document_result: Document text if match found, with source ID tags
        model: Configured Gemini GenerativeModel instance
        
    Returns:
        Dictionary with 'answer' and 'sources' keys, or fallback response
    """
    if not model:
        raise ValueError("Gemini model not provided")
    
    # Prepare tagged data for synthesis
    email_data_str = ""
    if email_results:
        email_summaries = []
        for email in email_results:
            source_id = email.get('_source_id', 'unknown')
            summary = f"[ID:{source_id}] From: {email.get('from', 'Unknown')}, Date: {email.get('date', 'Unknown')}, Subject: {email.get('subject', 'No subject')}, Body: {email.get('body', 'No content')[:300]}..."
            email_summaries.append(summary)
        email_data_str = "\n".join(email_summaries)
    else:
        email_data_str = "No relevant emails found."
    
    slack_data_str = ""
    if slack_results:
        slack_summaries = []
        for msg in slack_results:
            source_id = msg.get('_source_id', 'unknown')
            summary = f"[ID:{source_id}] Channel: {msg.get('channel', 'Unknown')}, User: {msg.get('user', 'Unknown')}, Date: {msg.get('date', 'Unknown')}, Message: {msg.get('message', 'No content')[:300]}..."
            slack_summaries.append(summary)
        slack_data_str = "\n".join(slack_summaries)
    else:
        slack_data_str = "No relevant Slack messages found."
    
    document_data_str = ""
    if document_result:
        # Extract document ID if present
        if "[DOCUMENT_ID:" in document_result:
            doc_id = document_result.split("[DOCUMENT_ID:")[1].split("]")[0]
            content = document_result.split("] ", 1)[1] if "] " in document_result else document_result
        else:
            doc_id = "unknown"
            content = document_result
        
        # Truncate document for synthesis if too long
        if len(content) > 1000:
            document_data_str = f"[ID:{doc_id}] {content[:1000]}..."
        else:
            document_data_str = f"[ID:{doc_id}] {content}"
    else:
        document_data_str = "No relevant document content found."
    
    # Create enhanced synthesis prompt for JSON output with citations
    prompt = f"""You are an AI assistant that provides accurate, well-sourced answers. Your task is to answer the user's query using ONLY the provided data sources.

User's Query: "{user_query}"

Available Data Sources:
- EMAILS: {email_data_str}
- SLACK MESSAGES: {slack_data_str}
- DOCUMENT: {document_data_str}

CRITICAL INSTRUCTIONS:
1. You MUST respond with a valid JSON object in this exact format:
   {{"answer": "your answer here with [1] [2] citations", "sources": [list of source objects]}}

2. In your answer string, place numbered citations [1], [2], [3], etc. after ANY claim that comes from the provided data.

3. For each citation [N], create a corresponding source object in the sources array with:
   - "id": the source ID number (from [ID:X] in the data)
   - "type": "email", "slack", or "document"
   - "content_snippet": a brief excerpt showing the relevant content
   - "metadata": additional details like sender, date, etc.

4. Base your answer ONLY on the provided data. Do not make up information.

5. If no relevant information is found, set answer to "No relevant information found in the provided data sources."

Example format:
{{"answer": "Project Phoenix received positive feedback [1] with some concerns about timeline [2].", "sources": [{{"id": 1, "type": "email", "content_snippet": "The technical approach looks solid", "metadata": {{"from": "sarah@company.com", "date": "2024-03-15"}}}}, {{"id": 3, "type": "slack", "content_snippet": "timeline seems aggressive", "metadata": {{"user": "alex", "channel": "project-phoenix"}}}}]}}

Respond with ONLY the JSON object, no additional text."""

    try:
        # Make API call to Gemini for synthesis
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2000,
                temperature=0.3,
                top_p=0.8
            )
        )
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini API")
        
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
            
            if "answer" not in parsed_response:
                raise ValueError("Response missing 'answer' field")
            
            if "sources" not in parsed_response:
                raise ValueError("Response missing 'sources' field")
            
            # Validate sources array
            sources = parsed_response["sources"]
            if not isinstance(sources, list):
                raise ValueError("'sources' field is not a list")
            
            # Validate each source object
            for i, source in enumerate(sources):
                if not isinstance(source, dict):
                    raise ValueError(f"Source {i} is not an object")
                if "id" not in source:
                    raise ValueError(f"Source {i} missing 'id' field")
                if "type" not in source:
                    raise ValueError(f"Source {i} missing 'type' field")
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            st.warning(f"Failed to parse JSON response: {e}")
            # Fallback to plain text response
            return {
                "answer": response_text,
                "sources": [],
                "_fallback": True,
                "_error": f"JSON parsing failed: {str(e)}"
            }
        
    except Exception as e:
        st.error(f"Error synthesizing results with citations: {e}")
        
        # Fallback: create a basic summary without AI synthesis
        fallback_response = {
            "answer": f"**Query:** {user_query}\n\n**Search Results:**\n",
            "sources": [],
            "_fallback": True,
            "_error": str(e)
        }
        
        if email_results:
            fallback_response["answer"] += f"**Found {len(email_results)} relevant emails:**\n"
            for i, email in enumerate(email_results[:3]):
                fallback_response["answer"] += f"• Email from {email.get('from', 'Unknown')} ({email.get('date', 'Unknown')}): {email.get('subject', 'No subject')}\n"
            if len(email_results) > 3:
                fallback_response["answer"] += f"• ... and {len(email_results) - 3} more emails\n"
            fallback_response["answer"] += "\n"
        
        if slack_results:
            fallback_response["answer"] += f"**Found {len(slack_results)} relevant Slack messages:**\n"
            for i, msg in enumerate(slack_results[:3]):
                fallback_response["answer"] += f"• Message from {msg.get('user', 'Unknown')} in #{msg.get('channel', 'Unknown')} ({msg.get('date', 'Unknown')})\n"
            if len(slack_results) > 3:
                fallback_response["answer"] += f"• ... and {len(slack_results) - 3} more messages\n"
            fallback_response["answer"] += "\n"
        
        if document_result:
            fallback_response["answer"] += "**Found relevant content in project documents**\n\n"
        
        if not email_results and not slack_results and not document_result:
            fallback_response["answer"] += "**No relevant information found** in emails, Slack messages, or documents for this query.\n"
        
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
    # Clear previous thoughts and start fresh
    clear_agent_thoughts()
    
    # Enhanced input validation
    is_valid, validation_error = validate_user_query(user_query)
    if not is_valid:
        add_agent_thought(
            "Input Validation", 
            user_query, 
            f"❌ Input Error: {validation_error}",
            {"validation_failed": True, "error_type": "input_validation"}
        )
        return f"❌ Input Error: {validation_error}"
    
    try:
        # Step 1: Load data
        try:
            email_data, slack_data, document_text = load_data()
            add_agent_thought(
                "Data Loading",
                "Loading emails.json, slack_messages.json, and project_notes.txt",
                {
                    "emails_loaded": len(email_data),
                    "slack_messages_loaded": len(slack_data),
                    "document_size": len(document_text)
                },
                {"step": 1, "status": "success"}
            )
        except Exception as e:
            add_agent_thought(
                "Data Loading",
                "Loading emails.json, slack_messages.json, and project_notes.txt",
                f"Error loading data: {e}",
                {"step": 1, "status": "error", "error_type": "data_loading"}
            )
            return f"Error loading data: {e}. Please ensure all data files are present and properly formatted."
        
        # Step 2: Configure Gemini API
        try:
            model = configure_gemini_api()
            add_agent_thought(
                "API Configuration",
                "Configuring Google Gemini API connection",
                "API configured successfully",
                {"step": 2, "status": "success", "model": "gemini-2.0-flash"}
            )
        except Exception as e:
            add_agent_thought(
                "API Configuration",
                "Configuring Google Gemini API connection",
                f"Error configuring Gemini API: {e}",
                {"step": 2, "status": "error", "error_type": "api_configuration"}
            )
            return f"Error configuring Gemini API: {e}. Please check your API key configuration."
        
        # Step 3: Generate search terms using Gemini
        try:
            search_terms = generate_search_terms(user_query, model)
            add_agent_thought(
                "Query Analysis",
                user_query,
                search_terms,
                {"step": 3, "status": "success", "method": "gemini_ai", "terms_count": len(search_terms)}
            )
            if not search_terms:
                add_agent_thought(
                    "Query Analysis",
                    user_query,
                    "Unable to generate search terms from your query",
                    {"step": 3, "status": "error", "error_type": "no_search_terms"}
                )
                return "Unable to generate search terms from your query. Please try rephrasing your question."
        except Exception as e:
            st.error(f"Search term generation failed: {e}")
            # Fallback: use original query words as search terms
            search_terms = [word.strip('.,!?') for word in user_query.split() if len(word) > 2][:3]
            add_agent_thought(
                "Query Analysis",
                user_query,
                search_terms,
                {"step": 3, "status": "fallback", "method": "fallback_extraction", "error": str(e)}
            )
            if not search_terms:
                add_agent_thought(
                    "Query Analysis",
                    user_query,
                    "Unable to process your query",
                    {"step": 3, "status": "error", "error_type": "fallback_failed"}
                )
                return "Unable to process your query. Please try rephrasing your question."
        
        # Step 4: Execute multi-source search using generated terms
        all_email_results = []
        all_slack_results = []
        document_result = ""
        
        try:
            # Search across all sources with each search term
            search_details = []
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
                
                # Track search results for this term
                search_details.append({
                    "term": term,
                    "email_matches": len(email_matches),
                    "slack_matches": len(slack_matches),
                    "document_match": bool(doc_match)
                })
            
            # Step 4.5: Tag search results with unique IDs for source chain
            tagged_emails, tagged_slack, tagged_document, source_mapping = tag_search_results_with_ids(
                all_email_results, all_slack_results, document_result
            )
            
            # Store source mapping in session state for UI display
            st.session_state.source_mapping = source_mapping
            
            add_agent_thought(
                "Multi-Source Search",
                search_terms,
                {
                    "total_email_results": len(all_email_results),
                    "total_slack_results": len(all_slack_results),
                    "document_found": bool(document_result),
                    "search_details": search_details,
                    "sources_tagged": len(source_mapping)
                },
                {"step": 4, "status": "success", "sources_searched": 3}
            )
                        
        except Exception as e:
            st.error(f"Search execution failed: {e}")
            add_agent_thought(
                "Multi-Source Search",
                search_terms,
                f"Error occurred while searching data sources: {e}",
                {"step": 4, "status": "error", "error_type": "search_execution"}
            )
            return f"Error occurred while searching data sources: {e}"
        
        # Step 5: Synthesize results using Gemini with source chain
        try:
            synthesized_response = synthesize_results_with_citations(
                user_query, 
                tagged_emails, 
                tagged_slack, 
                tagged_document, 
                model
            )
            add_agent_thought(
                "Result Synthesis with Citations",
                {
                    "query": user_query,
                    "email_results_count": len(tagged_emails),
                    "slack_results_count": len(tagged_slack),
                    "document_available": bool(tagged_document),
                    "sources_mapped": len(source_mapping)
                },
                synthesized_response,
                {"step": 5, "status": "success", "method": "gemini_ai_with_citations", "response_type": "json"}
            )
            
            # Store the structured response in session state for UI display
            st.session_state.final_response = synthesized_response
            
            return synthesized_response
            
        except Exception as e:
            st.error(f"Result synthesis failed: {e}")
            
            # Final fallback: return raw search results in structured format
            fallback_response = {
                "answer": f"**Search Results for:** {user_query}\n\n**Search Terms Used:** {', '.join(search_terms)}\n\n",
                "sources": [],
                "_fallback": True,
                "_error": str(e)
            }
            
            if all_email_results:
                fallback_response["answer"] += f"**Found {len(all_email_results)} relevant emails:**\n"
                for email in all_email_results[:3]:
                    fallback_response["answer"] += f"• {email.get('subject', 'No subject')} from {email.get('from', 'Unknown')}\n"
                if len(all_email_results) > 3:
                    fallback_response["answer"] += f"• ... and {len(all_email_results) - 3} more\n"
                fallback_response["answer"] += "\n"
            
            if all_slack_results:
                fallback_response["answer"] += f"**Found {len(all_slack_results)} relevant Slack messages:**\n"
                for msg in all_slack_results[:3]:
                    fallback_response["answer"] += f"• Message from {msg.get('user', 'Unknown')} in #{msg.get('channel', 'Unknown')}\n"
                if len(all_slack_results) > 3:
                    fallback_response["answer"] += f"• ... and {len(all_slack_results) - 3} more\n"
                fallback_response["answer"] += "\n"
            
            if document_result:
                fallback_response["answer"] += "**Found relevant content in project documents**\n\n"
            
            if not all_email_results and not all_slack_results and not document_result:
                fallback_response["answer"] += "**No relevant information found** for your query.\n"
            
            add_agent_thought(
                "Result Synthesis with Citations",
                {
                    "query": user_query,
                    "email_results_count": len(all_email_results),
                    "slack_results_count": len(all_slack_results),
                    "document_available": bool(document_result)
                },
                fallback_response,
                {"step": 5, "status": "fallback", "method": "fallback_summary", "error": str(e)}
            )
            
            # Store fallback response in session state
            st.session_state.final_response = fallback_response
            
            return fallback_response
            
    except Exception as e:
        st.error(f"Unexpected error in agent workflow: {e}")
        add_agent_thought(
            "Unexpected Error",
            user_query,
            f"An unexpected error occurred: {e}",
            {"step": "error", "status": "error", "error_type": "unexpected_error"}
        )
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
                    
                    # Show the agent's thinking process
                    show_agent_thinking()
                    
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
                    
                    # Display agent results using the new source chain feature
                    st.success("✅ Search completed!")
                    
                    # Check if we have a structured response with citations
                    if (hasattr(st.session_state, 'final_response') and 
                        isinstance(st.session_state.final_response, dict)):
                        
                        # Use the new source chain display
                        source_mapping = getattr(st.session_state, 'source_mapping', {})
                        display_answer_with_sources(st.session_state.final_response, source_mapping)
                        
                        # Show fallback warning if needed
                        if st.session_state.final_response.get('_fallback'):
                            st.warning("⚠️ This response was generated using fallback processing due to an error.")
                            if '_error' in st.session_state.final_response:
                                with st.expander("Error Details", expanded=False):
                                    st.code(st.session_state.final_response['_error'])
                    
                    else:
                        # Fallback to old display method for backward compatibility
                        st.write("### Results")
                        st.write(result)
                    
                    # Show the agent's thinking process
                    show_agent_thinking()
                    
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
            - Provides answers with clear source attribution and inline citations
            
            **Example queries:**
            - "What was the feedback on Project Phoenix?"
            - "Who mentioned the budget concerns?"
            - "What are the key points from the project notes?"
            - "Any messages about the deadline?"
            
            **Tips for better results:**
            - Be specific about what you're looking for
            - Use key terms that might appear in your data
            - Ask about specific people, projects, or topics
            
            **New Features:**
            - **Inline Citations**: Numbers like ¹ ² ³ in answers link to original sources
            - **Source Verification**: Click on source sections to see full original data
            - **Agent Thinking**: See how the AI processes your query step by step
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


if __name__ == "__main__":
    main()