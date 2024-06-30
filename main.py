import os
import sys
from crewai import Agent, Task, Crew, Process
from tools import report_tool, rag_tool
from agents import reportAgents
from tasks import reportTasks
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from custom_logger import get_logger
from custom_exceptions import log_exception

# Load environment variables
load_dotenv()

# Setup custom exception hook
sys.excepthook = log_exception

# Get the custom logger
logger = get_logger(__name__)

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

class OpenAIResponseModel(BaseModel):
    """Pydantic model for the OpenAI response."""
    is_generic: bool

class CrewManager:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def get_openai_response(self) -> OpenAIResponseModel:
        """Make an OpenAI API call and parse the response."""
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful test classification assistant. Dependent upon user query classify it into 'generic' or 'project specific'. A generic query is the one which is a generic question related to any topic e.g 'What are the financial options available in the docs?' and a project specific query is the one which has project specific question having detail of any specific project in the query"},
                {"role": "user", "content": "As per the following project guide me on the financial options: Marbury plaza project is a solar renovation project starting in the end of december 2024. State some financial options please"},
                {"role": "assistant", "content": "project specific"},
                {"role": "user", "content": self.prompt}
            ]
        )

        classification = response.choices[0].message.content.strip().lower()
        is_generic = classification == "generic"

        return OpenAIResponseModel(is_generic=is_generic)

    def start_crew(self, is_generic: bool) -> str:
        """Start the crew based on the query classification."""
        if is_generic:
            rag = rag_tool(self.prompt)
            rag_result = rag.qa_from_RAG()
            logger.info(f"RAG result: {rag_result}")
            return rag_result
        else:
            agents = reportAgents()
            tasks = reportTasks()

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
                memory=True
                )

            inputs = {'query': self.prompt}
            result = crew.kickoff(inputs=inputs)
            logger.info(f"Crew kickoff result: {result}")
            return result

# if __name__ == "__main__":
#     # Example prompt for testing
#     # Example usage
#     prompt = """Project Overview:

# The owner of Marbury Plaza wants to finance a comprehensive, deep energy retrofit of the property, including a 500kW solar project on the roof. The retrofit measures include:

# Envelope Improvements: Insulation, windows, air tightness, etc.

# Electric Lighting: Fixture upgrades, etc.

# Efficient Equipment

# Heating, Cooling & Ventilation

# Total Cost:

# Retrofit Measures: $50,000,000

# Rooftop Solar Project: $1,500,000 

# Total: $51,500,000

# Project Timeline:

# Start: January 2025

# Completion: July 2026

# Compliance Standards:

# Domestic sourcing of materials

# Prevailing wage requirements

# Davis-Bacon requirements

# Assumptions:

# The project will meet compliance standards related to workforce and materials.

# All necessary permits and approvals will be obtained in a timely manner.

# The owner is willing to explore and apply for various financing options."""

#     # Create CrewManager instance
#     manager = CrewManager(prompt)

#     # Get OpenAI response to determine if the query is generic or project-specific
#     openai_response = manager.get_openai_response()
#     logger.info(f"OpenAI response: {openai_response}")

#     # Start the crew based on the classification
#     result = manager.start_crew(openai_response.is_generic)
#     print(result)




