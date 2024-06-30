import os
from crewai import Task
from textwrap import dedent
from tools import report_tool

class reportTasks:
    def summary_task(self, agent, query):
        return Task(
            description=dedent(f"""\
                Analyze the project-specific query: {query}. Summarize the query into one statement,
                which includes all important details. It should compulsarily include each and every important financial figure,
                date, and any other project-related metric.
            """),
            expected_output=dedent("""\
                Your output should be a python List of single string containing the summary generated from the query.
            """),
            agent=agent
        )

    def policy_task(self, agent):
        return Task(
            description=dedent("""
                Use the output provided by summary_agent to extract policy from the tool. You should include the list provided by summary_agent in your input to the tool along with other keywords like
                compliance standards, Eligibility Criteria, Fees, Application review and selection process, etc.
                Example input is as under enclosed in HTML tag:
                <example_input>
                'Compliance standards, Eligibility criteria, Fees, Application review and selection process for a comprehensive energy retrofit,
                including a 500kW solar project on the roof. Retrofit measures involve envelope improvements, electric lighting upgrades, efficient equipment, and heating,
                cooling & ventilation. Total cost is $51,500,000, with a start date in January 2025 and completion in July 2026. Compliance standards include domestic sourcing of materials,
                prevailing wage requirements, and Davis-Bacon requirements. Assumptions are that the project will meet compliance standards, permits will be obtained timely, and the owner will explore financing options.'
                "The report tool will output a number of docs from the vectorstore"
                </example_input>
            """),
            expected_output=dedent("""
                A detailed and comprehensive document containing all the relevant eligibility criteria, compliance criteria, application procedure, fees about the project.
            """),
            tools=[report_tool()],
            agent=agent,
            )

    def financial_task(self, agent):
        return Task(
            description=dedent("""
                Use the output provided by summary_agent to extract financial options from the tool. You should include the list provided by summary_agent in your input to the tool along with other keywords like
                financial options, Subsidies, Grants, etc.
                Example input is enclosed in HTML tags:
                <example_input>
                Financial options, Subsidies, Grants for a comprehensive energy retrofit, including a 500kW solar project on the roof.
                Retrofit measures involve envelope improvements, electric lighting upgrades, efficient equipment, and heating, cooling & ventilation.
                Total cost is $51,500,000, with a start date in January 2025 and completion in July 2026. Compliance standards include domestic sourcing of materials,
                prevailing wage requirements, and Davis-Bacon requirements. Assumptions are that the project will meet compliance standards, permits will be obtained timely,
                and the owner will explore financing options.
                </example_input>
            """),
            expected_output=dedent("""
                Use the retrieved docs to formulate all the financial options, subsidies, grants, and their benefits related to the project.
            """),
            tools=[report_tool()],
            agent=agent
        )

    def report_task(self, agent):
        return Task(
            description=dedent("""
                Take the response out of policy agent and financial agent and make a detailed report on it.
            """),
            expected_output=dedent("""
                A structured report having proper headings, subheadings, and content.
            """),
            agent=agent
        )
