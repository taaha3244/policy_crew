import sys
from custom_logger import get_logger

# Get the custom logger
logger = get_logger(__name__)

def log_exception(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions with traceback."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Set the custom exception hook
sys.excepthook = log_exception

class CustomException(Exception):
    """Base class for custom exceptions."""
    def __init__(self, message):
        super().__init__(message)
        logger.error(message, exc_info=True)

class DataProcessingError(CustomException):
    """Exception raised for errors in the data processing."""
    def __init__(self, message="Error processing data"):
        super().__init__(message)

class RetrievalError(CustomException):
    """Exception raised for errors in the retrieval process."""
    def __init__(self, message="Error retrieving data"):
        super().__init__(message)
