import sys
import os
from typing import Dict
import yaml
from custom_logger import logger
from pydantic import BaseModel
from custom_exceptions import CustomException
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()



def get_hyperparameters_from_file():
    """
    Load hyperparameters from a YAML file.
    """
    try:
        # Correctly construct the path to hyper-parameters.yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        config_path = os.path.join(project_root, 'hyper-parameters.yaml')

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info("Hyperparameters loaded successfully.")
            return config
    except Exception as e:
        logger.error(f"Error loading hyperparameters: {e}")
        raise e
    

config=get_hyperparameters_from_file()

# Python class to differentiate between generic or project specific query
class OpenAIResponseModel(BaseModel):
    """Pydantic model for the OpenAI response."""
    is_generic: bool   

# Function to differentiate between project specific and generic query
def get_openai_response(prompt) -> OpenAIResponseModel:
    """
    Make an OpenAI API call to classify the query.

    Returns:
    OpenAIResponseModel: Model indicating if the query is generic.
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
        model=config['LLM_NAME'],
        messages=[
            {
            "role": "system",
            "content": (
                        "You are a helpful question classification assistant. You have the following tasks "
                        "1.Dependent upon user question classify it into 'generic' or 'project specific'. "
                        "Use these tips to classify:"
                        "1.A generic question is the one which is a generic question related to any topic "
                        "e.g 'What are the financial options available in the docs?'"
                        "2.A project specific question is the one related to ant specific project with some project related details"
                        "e.g Marbury project plaza is set to begin from april 2024. It is a detailed retrofit project in california which aims to install solar panels"
                        ),
                    },
                    {
            "role": "user",
            "content": (
                        "As per the following project guide me on the financial options: Al qasim project "
                        "is a building renovation project starting in the end of december 2024. State some financial options please"
                        ),
                    },
            {"role": "assistant", "content": "project specific"},
            {"role": "user", "content": prompt},
                ],
            )

        classification = response.choices[0].message.content.strip().lower()
        is_generic = classification == "generic"

        return OpenAIResponseModel(is_generic=is_generic)
    except Exception as e:
        logger.error(f"Error getting OpenAI response: {str(e)}")
        raise CustomException(f"Error getting OpenAI response: {e}", sys)
