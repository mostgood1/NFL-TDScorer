from app import create_app

# WSGI entrypoint for Gunicorn
app = create_app()
