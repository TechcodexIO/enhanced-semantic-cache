import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_CHAT_MODEL = "gpt-3.5-turbo"  # Added chat model
OPENAI_EMBEDDING_ENCODING = "cl100k_base"
OPENAI_MAX_TOKENS = 8191
OPENAI_EMBEDDING_DIMENSIONS = 1536

# Cache settings
CACHE_SIZE = 1000
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))

# CORS settings
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://localhost:8000")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
CORS_ALLOW_METHODS = os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE")
CORS_ALLOW_HEADERS = os.getenv("CORS_ALLOW_HEADERS", "*")

# Rate limiting settings
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/hour")

# Logging configuration
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
        },
    },
    'handlers': {
        'app_file': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOG_DIR, 'app.log'),
            'formatter': 'detailed',
        },
        'stress_test_file': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOG_DIR, 'stress_test.log'),
            'formatter': 'standard',
        },
        'api_metrics_file': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOG_DIR, 'api_metrics.log'),
            'formatter': 'standard',
        },
        'cache_metrics_file': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOG_DIR, 'cache_metrics.log'),
            'formatter': 'standard',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
        }
    },
    'loggers': {
        'stress_test': {
            'handlers': ['stress_test_file', 'console'],
            'level': 'INFO',
            'propagate': False
        },
        'api_metrics': {
            'handlers': ['api_metrics_file', 'console'],
            'level': 'INFO',
            'propagate': False
        },
        'models.cache_metrics': {
            'handlers': ['cache_metrics_file', 'console'],
            'level': 'INFO',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['app_file', 'console'],
        'level': 'INFO',
    },
}
