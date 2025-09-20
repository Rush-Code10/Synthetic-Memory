# Synthetic Memory Lite

AI-powered information retrieval and synthesis demo application that demonstrates agentic AI capabilities by intelligently searching through emails, chat messages, and documents to provide contextual answers using Google Gemini API.

## Overview

Synthetic Memory Lite showcases how AI agents can intelligently search across multiple data sources and synthesize information to answer user queries. The application uses Google's Gemini AI to generate relevant search terms and synthesize results with proper source attribution.

## Features

- **Multi-source Search**: Searches across emails, Slack messages, and documents
- **AI-powered Search Terms**: Uses Gemini AI to generate relevant search terms from natural language queries
- **Intelligent Synthesis**: Combines search results into coherent responses with source attribution
- **Error Handling**: Graceful handling of missing data, API failures, and invalid inputs
- **Performance Optimized**: Fast response times suitable for live demonstrations

## Architecture

The application follows a microservices-inspired architecture with clear separation of concerns:

- **Data Loading**: Handles JSON and text file parsing with validation
- **Search Engine**: Case-insensitive substring matching across all data sources
- **AI Integration**: Google Gemini API for search term generation and result synthesis
- **User Interface**: Streamlit web application with responsive design

## Data Sources

The application searches through three types of data:

1. **Emails** (`emails.json`): Email communications with from, date, subject, and body fields
2. **Slack Messages** (`slack_messages.json`): Chat messages with channel, user, date, and message fields
3. **Documents** (`project_notes.txt`): Plain text documentation and notes

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install streamlit google-generativeai
   ```
3. Set up your Gemini API key in `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```
4. Ensure data files are present: `emails.json`, `slack_messages.json`, `project_notes.txt`

## Usage

### Running the Application

```bash
streamlit run synthetic_memory.py
```

### Demo Query

The application comes pre-configured with a demo query:
"What was the feedback on Project Phoenix?"

This query demonstrates the complete workflow:
1. AI generates relevant search terms
2. System searches all data sources
3. Results are synthesized with proper source attribution

### Expected Output Format

```
Based on the data provided:
• Key finding 1
• Key finding 2
• Key finding 3

Sources:
• Email from user@company.com: Date - Description
• Slack message from user in #channel: Date - Description
• Document: Document name with relevant content
```

## Testing

The application includes comprehensive test coverage:

### Running Unit Tests
```bash
python run_tests.py
```

### Running Integration Tests
```bash
python test_integration.py
```

### Demo Validation
```bash
python demo_validation.py
```

### Test Coverage
- **Unit Tests**: 58 tests covering core functionality
- **Integration Tests**: 18 tests covering end-to-end workflows
- **Demo Validation**: 6 checks ensuring presentation readiness

## API Integration

### Google Gemini API

The application uses Google's Gemini Pro model for:
- **Search Term Generation**: Converting natural language queries into search terms
- **Result Synthesis**: Combining search results into coherent responses

### Configuration

API configuration includes:
- Automatic retry logic
- Timeout handling
- Quota management
- Error recovery with fallback processing

## Error Handling

The application provides graceful error handling for:
- Missing or corrupted data files
- API configuration issues
- Network connectivity problems
- Invalid user inputs
- AI service failures

All errors are presented with user-friendly messages and troubleshooting guidance.

## Performance

- **Data Loading**: < 0.1 seconds
- **Search Operations**: < 0.1 seconds
- **Complete Workflow**: < 5 seconds (typically ~0.2 seconds)
- **Memory Usage**: Optimized for large document processing

## Security

- Input validation prevents injection attacks
- API key management through Streamlit secrets
- No sensitive data exposure in error messages
- Secure handling of user queries

## Development

### Project Structure
```
synthetic-memory-lite/
├── synthetic_memory.py          # Main application
├── test_synthetic_memory.py     # Unit tests
├── test_integration.py          # Integration tests
├── demo_validation.py           # Demo validation script
├── run_tests.py                 # Test runner
├── emails.json                  # Sample email data
├── slack_messages.json          # Sample Slack data
├── project_notes.txt            # Sample document
└── README.md                    # This file
```

### Code Quality
- Comprehensive docstrings for all functions
- Type hints for better code maintainability
- Error handling with specific exception types
- Modular design for easy testing and extension

## Requirements

- Python 3.8+
- Streamlit 1.28+
- google-generativeai 0.3+
- Valid Google Gemini API key

## License

This project is a demonstration application for educational and hackathon purposes.

## Contributing

This is a demo application. For production use, consider:
- Database integration for larger datasets
- Advanced search algorithms (vector search, semantic search)
- User authentication and authorization
- Caching mechanisms for improved performance
- Advanced AI prompt engineering for better results