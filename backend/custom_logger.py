import logging
import os

def get_logger(name, log_file='app.log', console_level=logging.INFO, file_level=logging.ERROR):
    """
    Creates and returns a custom logger with specified name and log levels.
    
    Args:
        name (str): The name of the logger.
        log_file (str): The file path for the log file. Defaults to 'app.log'.
        console_level (int): The log level for console output. Defaults to logging.INFO.
        file_level (int): The log level for file output. Defaults to logging.ERROR.
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    # Check if the logger already has handlers to avoid duplicate logs
    if not logger.hasHandlers():
        # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger.setLevel(logging.DEBUG)

        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file)

        # Set log levels for handlers
        console_handler.setLevel(console_level)
        file_handler.setLevel(file_level)

        # Create formatters and add them to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

# Example usage:
# logger = get_logger(__name__, log_file='app.log', console_level=logging.INFO, file_level=logging.ERROR)
