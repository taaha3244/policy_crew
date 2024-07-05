import sys
from backend.custom_logger import get_logger

# Get the custom logger
logger = get_logger(__name__)

def log_exception(exc_type, exc_value, exc_traceback):
    """
    Log uncaught exceptions with traceback.
    
    Args:
        exc_type (type): Exception type.
        exc_value (Exception): Exception instance.
        exc_traceback (traceback): Traceback object.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Set the custom exception hook
sys.excepthook = log_exception

class CustomException(Exception):
    """
    Base class for custom exceptions.
    
    Args:
        message (str): Error message.
    """
    def __init__(self, message, *args):
        super().__init__(message, *args)
        logger.error(message, exc_info=True)

class DataProcessingError(CustomException):
    """
    Exception raised for errors in the data processing.
    
    Args:
        message (str): Error message. Defaults to "Error processing data".
    """
    def __init__(self, message="Error processing data", *args):
        super().__init__(message, *args)

class RetrievalError(CustomException):
    """
    Exception raised for errors in the retrieval process.
    
    Args:
        message (str): Error message. Defaults to "Error retrieving data".
    """
    def __init__(self, message="Error retrieving data", *args):
        super().__init__(message, *args)
