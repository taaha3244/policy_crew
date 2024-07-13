import os
from crewai import Task
from textwrap import dedent
from app.backend.tools import ReportTool

class ReportTasks:
    """
    A class to define tasks for agents in the RAG system.
    """

    def summary_task(self, agent, query):
        """
        Creates a summary task for the given query.

        Args:
            agent (Agent): The agent responsible for summarizing the query.
            query (str): The project-specific query.

        Returns:
            Task: The summary task.
        """
        return Task(
            description=dedent(f"""\
                Analyze the project-specific query: {query}. Summarize the query into one statement,
                which includes all important details. It should compulsorily include each and every important financial figure,
                date, and any other project-related metric.
            """),
            expected_output=dedent("""\
                Your output should be a Python list containing a single string with the summary generated from the query.
            """),
            agent=agent
        )

    def policy_task(self, agent):
        """
        Creates a policy extraction task.

        Args:
            agent (Agent): The agent responsible for extracting policy-related information.

        Returns:
            Task: The policy task.
        """
        return Task(
            description=dedent("""
                Use the output provided by summary_agent to extract policy information using the tool. Include the list provided by summary_agent in your input to the tool along with other keywords like
                compliance standards, eligibility criteria, fees, application review, and selection process, etc.
                Example input enclosed in HTML tags:
                <example_input>
                'Compliance standards, eligibility criteria, fees, application review and selection process for a comprehensive energy retrofit,
                including a 500kW solar project on the roof. Retrofit measures involve envelope improvements, electric lighting upgrades, efficient equipment, and heating,
                cooling & ventilation. Total cost is $51,500,000, with a start date in January 2025 and completion in July 2026. Compliance standards include domestic sourcing of materials,
                prevailing wage requirements, and Davis-Bacon requirements. Assumptions are that the project will meet compliance standards, permits will be obtained timely, and the owner will explore financing options.'
                "The report tool will output a number of docs from the vector store"
                </example_input>
            """),
            expected_output=dedent("""
                A detailed and comprehensive document containing all the relevant eligibility criteria, compliance criteria, and application procedure, fees about the project.
            """),
            tools=[ReportTool()],
            agent=agent,
        )

    def financial_task(self, agent):
        """
        Creates a financial options extraction task.

        Args:
            agent (Agent): The agent responsible for extracting financial options.

        Returns:
            Task: The financial task.
        """
        return Task(
            description=dedent("""
                Use the output provided by summary_agent to extract financial options using the tool. Include the list provided by summary_agent in your input to the tool along with other keywords like
                financial options, subsidies, grants, etc.
                Example input enclosed in HTML tags:
                <example_input>
                Financial options, subsidies, grants for a comprehensive energy retrofit, including a 500kW solar project on the roof.
                Retrofit measures involve envelope improvements, electric lighting upgrades, efficient equipment, and heating, cooling & ventilation.
                Total cost is $51,500,000, with a start date in January 2025 and completion in July 2026. Compliance standards include domestic sourcing of materials,
                prevailing wage requirements, and Davis-Bacon requirements. Assumptions are that the project will meet compliance standards, permits will be obtained timely,
                and the owner will explore financing options.
                </example_input>
            """),
            expected_output=dedent("""
                Use the retrieved docs to formulate all the financial options, subsidies, grants, and their benefits related to the project.
            """),
            tools=[ReportTool()],
            agent=agent
        )

    def report_task(self, agent):
        """
        Creates a report generation task.

        Args:
            agent (Agent): The agent responsible for generating the report.

        Returns:
            Task: The report task.
        """
        return Task(
            description=dedent("""
                Take the response from the policy agent and financial agent and make a detailed report on it.
            """),
            expected_output=dedent("""
                A structured report having proper headings, subheadings, and content.
            """),
            agent=agent
        )
