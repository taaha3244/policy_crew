from textwrap import dedent
from crewai import Agent
from backend.tools import ReportTool

class ReportAgents:
    """
    A class to define various agents for summarizing, policy extraction, financial data extraction, and report generation.
    """

    @staticmethod
    def summary_agent():
        """
        Creates an agent specialized in summarizing input into one statement with all relevant details.

        Returns:
            Agent: The summarizer agent.
        """
        return Agent(
            role='Expert Summarizer',
            goal='Summarize the given input into one statement, keeping all relevant and important details',
            verbose=True,
            memory=True,
            backstory=dedent("""\
                As an expert summarizer, your task is to summarize any given user input. 
                You are best at your work and have the ability to include each and every important detail into the summary.
            """),
            allow_delegation=False
        )

    @staticmethod
    def policy_agent():
        """
        Creates an agent specialized in extracting answers for policy-related queries.

        Returns:
            Agent: The policy expert agent.
        """
        return Agent(
            role='Policy Expert',
            goal='Extract answers for queries',
            verbose=True,
            memory=True,
            backstory=dedent("""\
                As a Policy Expert, you specialize in providing detailed and accurate 
                answers to generic policy-related questions. Your expertise helps ensure 
                that the user receives reliable information for their queries.
            """),
            tools=[ReportTool()],
            allow_delegation=False
        )

    @staticmethod
    def financial_agent():
        """
        Creates an agent specialized in extracting financial data for queries.

        Returns:
            Agent: The financial expert agent.
        """
        return Agent(
            role='Financial Expert',
            goal='Extract financial data for queries',
            verbose=True,
            memory=True,
            backstory=dedent("""\
                As a Financial Expert, you specialize in providing detailed and accurate 
                answers to generic finance questions. Your expertise helps ensure 
                that the user receives reliable information for their queries.
            """),
            tools=[ReportTool()],
            allow_delegation=False
        )

    @staticmethod
    def report_agent():
        """
        Creates an agent specialized in generating detailed reports from provided documents.

        Returns:
            Agent: The report analyst agent.
        """
        return Agent(
            role='Report Analyst',
            goal='Generate a detailed report out of the documents provided.',
            verbose=True,
            memory=True,
            backstory=dedent("""\
                As a Report Analyst, your role is to create a detailed report 
                based on the information provided to you by the question_generation agent.
                Convert the documents into proper headings and text for the reader to read.
            """),
            allow_delegation=False,
            output_file='Report.md'
        )
