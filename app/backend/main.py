import sys
import os
# Directly set the project root directory
project_root = "D:/policy_crew"
# Ensure the project root is at the top of sys.path
sys.path.insert(0, project_root)
from crewai import Crew, Process
from app.backend.tools import RAGTool, GraphRagTool
from app.backend.crewai_agent.agents import ReportAgents
from app.backend.crewai_agent.tasks import ReportTasks
from dotenv import load_dotenv
from custom_logger import logger
from custom_exceptions import CustomException
from app.backend.langgraph_agent.langraph import WorkflowManager
from app.backend.utils import get_hyperparameters_from_file 
from pydantic import BaseModel
from app.backend.utils import get_openai_response
# Load environment variables
load_dotenv()
# Loading hyper parameters from the yaml file
config= get_hyperparameters_from_file()
# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = config['LLM_NAME']

# Python class to differentiate between generic or project specific query
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







class LangraphManager:
    """
    Manages the conditional execution of RAG tool or Langraph workflow.

    Attributes:
        crew_manager (CrewManager): An instance of CrewManager.
    """

    def __init__(self, prompt: str):
        """
        Initializes the LangraphManager with a user prompt.

        Args:
            prompt (str): The user query or prompt.
        """
        self.openai_response=get_openai_response(prompt)
        self.prompt = prompt
        self.rag_tool = RAGTool(prompt)
        logger.info("LangraphManager initialized")

    def run_workflow(self) -> str:
        """
        Run the conditional workflow based on the query classification.

        Returns:
            str: The result of the workflow processing.
        """
        try:
            openai_response = self.openai_response
            logger.info(f"OpenAI response: {openai_response}")
            if openai_response.is_generic:
                rag = self.rag_tool
                rag_result = rag.qa_from_RAG()
                logger.info(f"RAG result: {rag_result}")
                return rag_result
            else:
                return self.run_langraph_workflow()
        except Exception as e:
            logger.error(f"Error running conditional workflow: {str(e)}")
            raise CustomException(f"Error running conditional workflow: {e}", sys)

    def run_langraph_workflow(self) -> str:
        """
        Run the langraph workflow.

        Returns:
            str: The result of the langraph workflow.
        """
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            workflow_manager = WorkflowManager(openai_api_key)
            result = workflow_manager.run(self.prompt)
            if result:
                logger.info(f"Langraph workflow result: {result}")
                return result
            else:
                raise ValueError("Langraph workflow returned None")
        except Exception as e:
            logger.error(f"Error running langraph workflow: {str(e)}")
            raise CustomException(f"Error running langraph workflow: {e}", sys)

