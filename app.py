"""
Synthetic Memory Lite - Flask Web Application

AI-powered information retrieval and synthesis with professional UI design.
Converts Streamlit app to Flask with modern animations and effects.
"""

from flask import Flask, render_template, request, jsonify, session, send_file
from flask_session import Session
import json
import os
import tempfile
import base64
import io
from datetime import datetime
from typing import List, Dict, Tuple
import google.generativeai as genai
from gtts import gTTS
import re

# Import core functions from synthetic_memory
from synthetic_memory import (
    load_data,
    search_emails,
    search_slack,
    search_documents,
    tag_search_results_with_ids,
    validate_user_query,
    get_available_voices,
    clean_text_for_tts
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'synthetic-memory-lite-2024'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = tempfile.mkdtemp()
Session(app)

# Global variables for data and model
email_data = []
slack_data = []
document_text = ""
gemini_model = None

# Flask-compatible versions of Streamlit-dependent functions
def configure_gemini_api_flask():
    """Configure Gemini API for Flask (without Streamlit)"""
    try:
        # Read API key from secrets.toml
        import toml
        secrets_path = '.streamlit/secrets.toml'
        
        if os.path.exists(secrets_path):
            with open(secrets_path, 'r') as f:
                secrets = toml.load(f)
                api_key = secrets.get('GEMINI_API_KEY')
        else:
            # Fallback to environment variable
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in secrets.toml or environment variables")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        return model
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        raise

def generate_search_terms_flask(user_query: str, model: genai.GenerativeModel) -> List[str]:
    """Generate search terms using Gemini (Flask version)"""
    if not user_query or not isinstance(user_query, str):
        return []
    
    if not model:
        raise ValueError("Gemini model not provided")
    
    prompt = f"""Your role is to analyze a user's query and determine the best keywords to search for in a database of emails, chat messages, and documents.

User Query: "{user_query}"

Please generate a simple JSON object containing a list of the most relevant and specific search strings. Focus on key terms, names, and concepts. Limit to 5 terms maximum. Only respond with the JSON, nothing else.

Format: {{ "search_terms": ["term1", "term2", "term3"] }}"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.3,
                top_p=0.8
            )
        )
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini API")
        
        response_text = response.text.strip()
        
        # Clean up response text
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        # Parse JSON response
        try:
            parsed_response = json.loads(response_text)
            
            if not isinstance(parsed_response, dict) or "search_terms" not in parsed_response:
                raise ValueError("Invalid response format")
            
            search_terms = parsed_response["search_terms"]
            
            if not isinstance(search_terms, list):
                raise ValueError("'search_terms' field is not a list")
            
            # Filter and validate search terms
            valid_terms = []
            for term in search_terms:
                if isinstance(term, str) and term.strip():
                    clean_term = term.strip()
                    if len(clean_term) > 1 and clean_term.lower() not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                        valid_terms.append(clean_term)
            
            return valid_terms[:5]
            
        except json.JSONDecodeError:
            # Fallback: extract key words from original query
            words = user_query.split()
            return [word.strip('.,!?') for word in words if len(word) > 2][:3]
        
    except Exception as e:
        print(f"Error generating search terms: {e}")
        # Fallback: extract key words from original query
        words = user_query.split()
        return [word.strip('.,!?') for word in words if len(word) > 2][:3]

def synthesize_results_with_citations_flask(user_query: str, email_results: List[Dict], slack_results: List[Dict], 
                                         document_result: str, model: genai.GenerativeModel) -> Dict:
    """Synthesize results with citations (Flask version)"""
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
        if "[DOCUMENT_ID:" in document_result:
            doc_id = document_result.split("[DOCUMENT_ID:")[1].split("]")[0]
            content = document_result.split("] ", 1)[1] if "] " in document_result else document_result
        else:
            doc_id = "unknown"
            content = document_result
        
        if len(content) > 1000:
            document_data_str = f"[ID:{doc_id}] {content[:1000]}..."
        else:
            document_data_str = f"[ID:{doc_id}] {content}"
    else:
        document_data_str = "No relevant document content found."
    
    # Create synthesis prompt
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

Respond with ONLY the JSON object, no additional text."""

    try:
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
        
        # Clean up response text
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        # Parse JSON response
        try:
            parsed_response = json.loads(response_text)
            
            if not isinstance(parsed_response, dict) or "answer" not in parsed_response:
                raise ValueError("Invalid response format")
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return {
                "answer": response_text,
                "sources": [],
                "_fallback": True,
                "_error": f"JSON parsing failed: {str(e)}"
            }
        
    except Exception as e:
        print(f"Error synthesizing results: {e}")
        
        # Fallback response
        fallback_response = {
            "answer": f"**Search Results for:** {user_query}\n\n**Search Results:**\n",
            "sources": [],
            "_fallback": True,
            "_error": str(e)
        }
        
        if email_results:
            fallback_response["answer"] += f"**Found {len(email_results)} relevant emails:**\n"
            for email in email_results[:3]:
                fallback_response["answer"] += f"• {email.get('subject', 'No subject')} from {email.get('from', 'Unknown')}\n"
            if len(email_results) > 3:
                fallback_response["answer"] += f"• ... and {len(email_results) - 3} more\n"
            fallback_response["answer"] += "\n"
        
        if slack_results:
            fallback_response["answer"] += f"**Found {len(slack_results)} relevant Slack messages:**\n"
            for msg in slack_results[:3]:
                fallback_response["answer"] += f"• Message from {msg.get('user', 'Unknown')} in #{msg.get('channel', 'Unknown')}\n"
            if len(slack_results) > 3:
                fallback_response["answer"] += f"• ... and {len(slack_results) - 3} more\n"
            fallback_response["answer"] += "\n"
        
        if document_result:
            fallback_response["answer"] += "**Found relevant content in project documents**\n\n"
        
        if not email_results and not slack_results and not document_result:
            fallback_response["answer"] += "**No relevant information found** for your query.\n"
        
        return fallback_response

def text_to_speech_flask(text: str, voice_name: str = None) -> str:
    """Convert text to speech using gTTS (Flask version)"""
    if not text or not text.strip():
        return None
    
    # Get voice configuration
    voices = get_available_voices()
    if voice_name is None:
        voice_name = 'English (US) - Female'
    
    voice_config = voices.get(voice_name, voices['English (US) - Female'])
    
    try:
        # Clean text for better TTS pronunciation
        cleaned_text = clean_text_for_tts(text)
        
        # Create gTTS object
        tts = gTTS(
            text=cleaned_text,
            lang=voice_config['lang'],
            tld=voice_config['tld'],
            slow=voice_config.get('slow', False)
        )
        
        # Generate audio in memory
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Convert to base64 for web
        audio_data = audio_buffer.getvalue()
        audio_base64 = base64.b64encode(audio_data).decode()
        
        return f"data:audio/mp3;base64,{audio_base64}"
        
    except Exception as e:
        print(f"Text-to-speech conversion failed: {e}")
        return None

def initialize_app():
    """Initialize the application with data and model"""
    global email_data, slack_data, document_text, gemini_model
    
    try:
        # Load data
        email_data, slack_data, document_text = load_data()
        
        # Configure Gemini API
        gemini_model = configure_gemini_api_flask()
        
        return True
    except Exception as e:
        print(f"Initialization error: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Check application status"""
    return jsonify({
        'status': 'ready',
        'data_loaded': len(email_data) > 0,
        'model_ready': gemini_model is not None,
        'emails_count': len(email_data),
        'slack_count': len(slack_data),
        'document_size': len(document_text)
    })

@app.route('/api/voices')
def api_voices():
    """Get available TTS voices"""
    return jsonify(get_available_voices())

@app.route('/api/query', methods=['POST'])
def api_query():
    """Process user query and return results"""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Validate query
        is_valid, validation_error = validate_user_query(user_query)
        if not is_valid:
            return jsonify({'error': validation_error}), 400
        
        # Generate search terms
        search_terms = generate_search_terms_flask(user_query, gemini_model)
        if not search_terms:
            return jsonify({'error': 'Unable to generate search terms'}), 500
        
        # Search across all sources
        all_email_results = []
        all_slack_results = []
        document_result = ""
        
        for term in search_terms:
            # Search emails
            email_matches = search_emails(term, email_data)
            for email in email_matches:
                if email not in all_email_results:
                    all_email_results.append(email)
            
            # Search Slack messages
            slack_matches = search_slack(term, slack_data)
            for msg in slack_matches:
                if msg not in all_slack_results:
                    all_slack_results.append(msg)
            
            # Search documents
            if not document_result:
                doc_match = search_documents(term, document_text)
                if doc_match:
                    document_result = doc_match
        
        # Tag search results with IDs
        tagged_emails, tagged_slack, tagged_document, source_mapping = tag_search_results_with_ids(
            all_email_results, all_slack_results, document_result
        )
        
        # Synthesize results
        synthesized_response = synthesize_results_with_citations_flask(
            user_query, tagged_emails, tagged_slack, tagged_document, gemini_model
        )
        
        # Add source mapping to response
        synthesized_response['source_mapping'] = source_mapping
        
        return jsonify({
            'success': True,
            'query': user_query,
            'search_terms': search_terms,
            'response': synthesized_response,
            'stats': {
                'email_results': len(all_email_results),
                'slack_results': len(all_slack_results),
                'document_found': bool(document_result)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Query processing failed: {str(e)}'}), 500

@app.route('/api/tts', methods=['POST'])
def api_tts():
    """Generate text-to-speech audio"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        voice = data.get('voice', 'English (US) - Female')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Generate audio
        audio_data = text_to_speech_flask(text, voice)
        
        if not audio_data:
            return jsonify({'error': 'TTS generation failed'}), 500
        
        # Return base64 audio data
        return jsonify({
            'success': True,
            'audio_data': audio_data,
            'voice': voice
        })
        
    except Exception as e:
        return jsonify({'error': f'TTS generation failed: {str(e)}'}), 500

@app.route('/api/tts/preview', methods=['POST'])
def api_tts_preview():
    """Generate TTS preview for voice testing"""
    try:
        data = request.get_json()
        voice = data.get('voice', 'English (US) - Female')
        
        preview_text = "Hello! This is a preview of the selected voice. The text-to-speech feature will read your responses aloud."
        
        # Generate audio
        audio_data = text_to_speech_flask(preview_text, voice)
        
        if not audio_data:
            return jsonify({'error': 'Preview generation failed'}), 500
        
        return jsonify({
            'success': True,
            'audio_data': audio_data,
            'voice': voice
        })
        
    except Exception as e:
        return jsonify({'error': f'Preview generation failed: {str(e)}'}), 500

@app.route('/api/agent-thoughts')
def api_agent_thoughts():
    """Get agent thinking process (placeholder for future implementation)"""
    return jsonify({
        'thoughts': [],
        'message': 'Agent thinking process will be implemented in future version'
    })

if __name__ == '__main__':
    # Initialize the application
    if initialize_app():
        print("✅ Application initialized successfully")
        print("🚀 Starting Flask server...")
        # Use PORT environment variable if available (for Render)
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("❌ Application initialization failed")
        print("Please check your configuration and try again")
