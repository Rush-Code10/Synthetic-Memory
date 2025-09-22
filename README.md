# Synthetic Memory Lite - Flask Web Application

A professional, modern web application for AI-powered information retrieval and synthesis with advanced text-to-speech capabilities.

## Features

### Core Functionality
- **Multi-source Search**: Intelligently searches across emails, Slack messages, and documents
- **AI-Powered Analysis**: Uses Google Gemini AI for query understanding and result synthesis
- **Source Attribution**: Provides detailed citations and source verification
- **Real-time Processing**: Fast, responsive search with live status updates

### Text-to-Speech (TTS)
- **12 High-Quality Voices**: Multiple English accents (US, UK, AU, CA, IN, IE, ZA, NZ)
- **Voice Preview**: Test voices before generating full responses
- **Speed Control**: Adjustable speech speed from 0.5x to 2.0x
- **Audio Download**: Download responses as MP3 files
- **Smart Text Processing**: Automatically optimizes text for natural speech

### Professional UI/UX
- **Modern Design**: Clean, professional interface with smooth animations
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **Interactive Elements**: Hover effects, loading states, and smooth transitions
- **Real-time Feedback**: Live status indicators and progress tracking
- **Accessibility**: Keyboard navigation and screen reader support

## Design Features

### Visual Elements
- **Gradient Backgrounds**: Beautiful color gradients throughout the interface
- **Smooth Animations**: Fade-in, slide-up, and hover effects
- **Professional Typography**: Inter font family for optimal readability
- **Color-coded Status**: Visual indicators for different states
- **Card-based Layout**: Clean, organized information presentation

### Interactive Components
- **Animated Loading Screen**: Professional startup experience
- **Dynamic Status Indicators**: Real-time system status updates
- **Expandable Panels**: Collapsible sections for better organization
- **Modal Dialogs**: Clean popup interfaces for audio playback
- **Toast Notifications**: Non-intrusive success/error messages

## Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd synthetic-memory
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_flask.txt
   ```

3. **Configure API key**
   ```bash
   # Create .streamlit/secrets.toml
   echo 'GEMINI_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

### Manual Installation

```bash
pip install flask flask-session google-generativeai gtts toml
```

## Project Structure

```
synthetic-memory/
├── app.py                          # Main Flask application
├── synthetic_memory.py             # Core business logic
├── requirements_flask.txt          # Flask dependencies
├── templates/
│   └── index.html                  # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css              # Professional CSS with animations
│   └── js/
│       └── app.js                 # Interactive JavaScript
├── .streamlit/
│   └── secrets.toml               # API configuration
├── emails.json                    # Sample email data
├── slack_messages.json            # Sample Slack data
├── project_notes.txt              # Sample document
└── test_flask_app.py              # Test script
```

## Usage

### Basic Search
1. Enter your question in the search box
2. Click "Find Context" or press Enter
3. View AI-generated results with source citations
4. Use TTS to listen to responses (optional)

### Text-to-Speech
1. Expand "Text-to-Speech Settings"
2. Select your preferred voice
3. Adjust speech speed if needed
4. Click "Preview Voice" to test
5. Enable TTS for automatic audio generation

### Quick Queries
Use the pre-built quick query buttons:
- "Project Phoenix Feedback"
- "Budget Concerns"
- "Project Notes"

## API Endpoints

### Main Endpoints
- `GET /` - Main application page
- `GET /api/status` - Application status and data counts
- `GET /api/voices` - Available TTS voices

### Query Endpoints
- `POST /api/query` - Process search queries
- `POST /api/tts` - Generate text-to-speech audio
- `POST /api/tts/preview` - Preview voice samples

### Request/Response Format

#### Query Request
```json
{
  "query": "What was the feedback on Project Phoenix?",
  "tts_enabled": true,
  "voice": "English (US) - Female"
}
```

#### Query Response
```json
{
  "success": true,
  "query": "What was the feedback on Project Phoenix?",
  "search_terms": ["Project Phoenix", "feedback", "review"],
  "response": {
    "answer": "Based on the data provided...",
    "sources": [
      {
        "id": 1,
        "type": "email",
        "content_snippet": "The technical approach looks solid...",
        "metadata": {
          "from": "sarah.chen@company.com",
          "date": "2024-03-15"
        }
      }
    ]
  },
  "stats": {
    "email_results": 6,
    "slack_results": 6,
    "document_found": true
  }
}
```

## Customization

### Styling
Modify `static/css/style.css` to customize:
- Color scheme (CSS variables in `:root`)
- Animations and transitions
- Layout and spacing
- Typography and fonts

### Functionality
Extend `static/js/app.js` to add:
- New interactive features
- Additional API endpoints
- Custom animations
- Enhanced user experience

### Backend
Modify `app.py` to add:
- New API endpoints
- Additional data sources
- Enhanced processing logic
- Custom error handling

## Testing

### Run Tests
```bash
python test_flask_app.py
```

### Test Coverage
- Application startup
- API endpoints
- Query processing
- TTS functionality
- Error handling

## Deployment

### Production Setup
1. **Environment Variables**
   ```bash
   export GEMINI_API_KEY="your-api-key"
   export FLASK_ENV="production"
   ```

2. **WSGI Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Docker** (optional)
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements_flask.txt
   EXPOSE 5000
   CMD ["python", "app.py"]
   ```

## Security

- **Input Validation**: All user inputs are validated and sanitized
- **API Key Protection**: Secure storage in configuration files
- **Error Handling**: Graceful error handling without information leakage
- **Rate Limiting**: Built-in protection against abuse

## Performance

- **Fast Response Times**: < 2 seconds for typical queries
- **Efficient Processing**: Optimized search algorithms
- **Caching**: Session-based caching for improved performance
- **Responsive UI**: Smooth animations without performance impact

## Key Improvements Over Streamlit

1. **Professional UI**: Modern, polished interface with animations
2. **Better Performance**: Faster loading and more responsive
3. **Mobile Support**: Fully responsive design
4. **Enhanced TTS**: Better voice selection and controls
5. **Real-time Feedback**: Live status updates and progress indicators
6. **Modular Architecture**: Clean separation of concerns
7. **API-First Design**: Easy to extend and integrate
8. **Production Ready**: Proper error handling and logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and demonstration purposes.

## Support

For issues and questions:
1. Check the test results: `python test_flask_app.py`
2. Verify API key configuration
3. Check browser console for JavaScript errors
4. Review Flask application logs

---

**Synthetic Memory Lite** - Transforming how you interact with your personal data through AI-powered search and synthesis.
