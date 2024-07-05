import os
import pymongo
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_database():
    """
    Establish a connection to the MongoDB database.

    Returns:
        db: A reference to the MongoDB database.
    """
    try:
        mongodb_url = os.getenv('MONGODB_URL')
        logger.debug(f"Connecting to MongoDB with URL: {mongodb_url}")
        if not mongodb_url:
            raise ValueError("MONGODB_URL environment variable is not set.")
        
        client = pymongo.MongoClient(mongodb_url)
        logger.info('Successfully connected to MongoDB')
        
        # Use a database named "myDatabase"
        db = client.myDatabase
        return db
    
    except pymongo.errors.ConfigurationError as e:
        logger.error("Invalid URI host error:", exc_info=True)
        raise ValueError("An Invalid URI host error was received. Check your Atlas host name in the connection string.")
    
    except pymongo.errors.ConnectionError as e:
        logger.error("MongoDB connection error:", exc_info=True)
        raise ValueError("Unable to connect to MongoDB. Check your network connection and MongoDB configuration.")
    
    except Exception as e:
        logger.error("Unexpected error:", exc_info=True)
        raise ValueError(f"An unexpected error occurred: {e}")

def insert_sample_document():
    try:
        db = get_database()
        questions_collection = db["questions"]
        
        sample_document = {
            "question": "What is the capital of France?",
            "response": "Paris"
        }
        
        result = questions_collection.insert_one(sample_document)
        if result.acknowledged:
            logger.info(f"Document inserted with ID: {result.inserted_id}")
        else:
            logger.error("Failed to insert document.")
    except Exception as e:
        logger.error(f"Error inserting document: {e}")

# Example usage
if __name__ == "__main__":
    insert_sample_document()
