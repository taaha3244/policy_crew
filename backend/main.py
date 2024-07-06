import sys
import os
from crewai import Crew, Process
from backend.tools import RAGTool
from backend.agents import ReportAgents
from backend.tasks import ReportTasks
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from backend.custom_logger import logger
from backend.custom_exceptions import CustomException

# Load environment variables
load_dotenv()

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"

class OpenAIResponseModel(BaseModel):
    """Pydantic model for the OpenAI response."""
    is_generic: bool

class CrewManager:
    """
    Manages the crew of agents and processes queries using OpenAI's API.

    Attributes:
        prompt (str): The user query or prompt.
    """

    def __init__(self, prompt: str):
        """
        Initializes the CrewManager with a user prompt.

        Args:
            prompt (str): The user query or prompt.
        """
        self.prompt = prompt

    def get_openai_response(self) -> OpenAIResponseModel:
        """
        Make an OpenAI API call to classify the query.

        Returns:
            OpenAIResponseModel: Model indicating if the query is generic.
        """
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful test classification assistant. "
                            "Dependent upon user query classify it into 'generic' or 'project specific'. "
                            "A generic query is the one which is a generic question related to any topic "
                            "e.g 'What are the financial options available in the docs?' and a project specific query "
                            "is the one which has project specific question having detail of any specific project in the query"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "As per the following project guide me on the financial options: Marbury plaza project "
                            "is a solar renovation project starting in the end of december 2024. State some financial options please"
                        ),
                    },
                    {"role": "assistant", "content": "project specific"},
                    {"role": "user", "content": self.prompt},
                ],
            )

            classification = response.choices[0].message.content.strip().lower()
            is_generic = classification == "generic"

            return OpenAIResponseModel(is_generic=is_generic)
        except Exception as e:
            logger.error(f"Error getting OpenAI response: {str(e)}")
            raise CustomException(f"Error getting OpenAI response: {e}", sys)

    def start_crew(self, is_generic: bool) -> str:
        """
        Start the crew based on the query classification.

        Args:
            is_generic (bool): Indicates if the query is generic.

        Returns:
            str: The result of the crew processing.
        """
        try:
            if is_generic:
                rag = RAGTool(self.prompt)
                rag_result = rag.qa_from_RAG()
                logger.info(f"RAG result: {rag_result}")
                return rag_result
            else:
                agents = ReportAgents()
                tasks = ReportTasks()

                # Create Agents
                summary_agent = agents.summary_agent()
                policy_agent = agents.policy_agent()
                financial_agent = agents.financial_agent()
                report_agent = agents.report_agent()

                # Create Tasks
                summary_task = tasks.summary_task(summary_agent, self.prompt)
                policy_task = tasks.policy_task(policy_agent)
                financial_task = tasks.financial_task(financial_agent)
                report_task = tasks.report_task(report_agent)

                # Form the crew
                crew = Crew(
                    agents=[summary_agent, policy_agent, financial_agent, report_agent],
                    tasks=[summary_task, policy_task, financial_task, report_task],
                    process=Process.sequential,
                    verbose=True,
                    memory=True,
                )

                inputs = {"query": self.prompt}
                result = crew.kickoff(inputs=inputs)
                logger.info(f"Crew kickoff result: {result}")
                return result
        except Exception as e:
            logger.error(f"Error starting crew: {str(e)}")
            raise CustomException(f"Error starting crew: {e}", sys)


# Uncomment for standalone testing
# if __name__ == "__main__":
#     prompt = "Your test prompt here"
#     manager = CrewManager(prompt)
#     openai_response = manager.get_openai_response()
#     logger.info(f"OpenAI response: {openai_response}")
#     result = manager.start_crew(openai_response.is_generic)
#     print(result)
