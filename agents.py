from textwrap import dedent
import os
from crewai import Agent

from tools import report_tool, rag_tool

# Define Crew Class for Agents

class reportAgents:


  def summary_agent(self):
    return Agent(
    role='Expert Summarizer',
    goal='Summarize the given input into one statement, keeping all relevant and important details',
    verbose=True,
    memory=True,
    backstory=dedent("""\
        As an expert summarizer , your task is to summarize any given user input 
        You are best at your work and have the ability to include each and every important detail into the summary 
        """
    ),
    allow_delegation=False
)

  def policy_agent(self):
    return Agent(
    role='Policy Expert',
    goal='Extract answers for queries',
    verbose=True,
    memory=True,
    backstory=dedent("""
        As a Policy Expert, you specialize in providing detailed and accurate 
        answers to generic policy-related questions. Your expertise helps ensure 
        that the user receives reliable information for their  queries."""
    ),
    tools=[report_tool()],
    allow_delegation=False,
)
    
  def financial_agent(self):
    return Agent(
    role='Financial Expert',
    goal='Extract financial data for queries',
    verbose=True,
    memory=True,
    backstory=dedent("""
        As a Financial, you specialize in providing detailed and accurate 
        answers to generic finance questions. Your expertise helps ensure 
        that the user receives reliable information for their  queries."""
    ),
    tools=[report_tool()],
    allow_delegation=False
    )
  def report_agent(self):
    return Agent(
    role='Report Analyst',
    goal='Generate a detailed report out of the Documents provided. ',
    verbose=True,
    memory=True,
    backstory=dedent("""        
        As a Report Analyst, your role is create a detailed report 
        based on the information provided to you by the question_generation agent
        Convert the documents in to proper Headings and Text for the reader to read"""
    ),
    allow_delegation=False,
    output_file='Report.md'
)
