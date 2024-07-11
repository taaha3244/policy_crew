import sys
import os
sys.path.append('D:/POLICY_CREW')
from crewai import Crew, Process
from backend.tools import RAGTool, GraphRagTool
from backend.agents import ReportAgents
from backend.tasks import ReportTasks
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from backend.custom_logger import logger
from backend.custom_exceptions import CustomException
from backend.langraph import WorkflowManager 

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
        self.crew_manager = CrewManager(prompt)
        self.prompt = prompt
        self.rag_tool=RAGTool(prompt)
        self.graph_rag_tool=GraphRagTool(prompt)

    def run_workflow(self) -> str:
        """
        Run the conditional workflow based on the query classification.

        Returns:
            str: The result of the workflow processing.
        """
        try:
            openai_response = self.crew_manager.get_openai_response()
            if openai_response.is_generic:
                rag=self.graph_rag_tool
                rag_result = rag.load_neo4j_graph()
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
       



# Uncomment for standalone testing
# if __name__ == "__main__":
#     prompt = "Your test prompt here"
#     manager = CrewManager(prompt)
#     openai_response = manager.get_openai_response()
#     logger.info(f"OpenAI response: {openai_response}")
#     result = manager.start_crew(openai_response.is_generic)
#     print(result)


# Uncomment for standalone testing
# if __name__ == "__main__":
#     try:
#         prompt = "Make a detailed report on a solar retrofit project including well fare for elders."
#         langraph_manager = LangraphManager(prompt)
#         result = langraph_manager.run_workflow()
#         print(result)
#     except Exception as e:
#         logger.error("Error in main execution.")
#         raise CustomException(e, sys)