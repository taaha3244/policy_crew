import os
import redis
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = int(os.getenv('REDIS_PORT'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
REDIS_DB = int(os.getenv('REDIS_DB', 0))

def get_redis_client():
    """
    Establish a connection to the Redis database.

    Returns:
        r: A reference to the Redis client.
    """
    try:
        logger.debug(f"Connecting to Redis with HOST: {REDIS_HOST}, PORT: {REDIS_PORT}, DB: {REDIS_DB}")
        if not REDIS_HOST or not REDIS_PORT or not REDIS_PASSWORD:
            raise ValueError("Redis connection details are not set in environment variables.")
        
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD
        )
        # Test the connection
        r.ping()
        logger.info('Successfully connected to Redis')
        
        return r
    
    except redis.ConnectionError as e:
        logger.error("Redis connection error:", exc_info=True)
        raise ValueError("Unable to connect to Redis. Check your network connection and Redis configuration.")
    
    except Exception as e:
        logger.error("Unexpected error:", exc_info=True)
        raise ValueError(f"An unexpected error occurred: {e}")

# Get Redis client
redis_client = get_redis_client()

# Example usage
if __name__ == "__main__":
    try:
        redis_client = get_redis_client()
        # Example Redis operations
        redis_client.set('key', 'value')
        print(redis_client.get('key'))  # Output: b'value'
    except Exception as e:
        logger.error(f"Failed to use Redis client: {e}")

__all__ = ["redis_client"]
