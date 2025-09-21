"""
WSGI Entry Point for Render Deployment
"""

from app import app, initialize_app

# Initialize the application when the module is imported
if initialize_app():
    print("✅ Application initialized successfully for production")
else:
    print("❌ Application initialization failed")
    raise RuntimeError("Failed to initialize application")

# This is what gunicorn will use
if __name__ == "__main__":
    app.run()
