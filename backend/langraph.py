import os
import sys
from dotenv import load_dotenv
from typing import List, Optional, Type, Literal
from qdrant_client import QdrantClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.qdrant import Qdrant
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
import functools
from langchain_core.messages import AIMessage
import operator
from typing import Annotated, Sequence, TypedDict

# Import custom logger and exceptions
from backend.custom_logger import logger
from backend.custom_exceptions import CustomException

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

class WorkflowManager:
    def __init__(self, openai_api_key: str):
        try:
            self.openai_api_key = openai_api_key
            self.llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
            self.report_tool_instance = self._create_report_tool()
            self.workflow = StateGraph(AgentState)
            self._setup_workflow()
            self.graph = self.workflow.compile()
            logger.info("WorkflowManager initialized successfully.")
        except Exception as e:
            logger.error("Error during WorkflowManager initialization.")
            raise CustomException(e, sys)

    def _create_report_tool(self):
        try:
            class ReportToolInput(BaseModel):
                query: List[str] = Field(description="A list of inputs for the RAG pipeline")

            class ReportTool(BaseTool):
                name: str = "report_tool"
                description: str = "Tool to retrieve relevant documents from the vector database using a list of user queries and return a response."
                args_schema: Optional[Type[BaseModel]] = ReportToolInput
                return_direct: bool = True

                def _run(self, query: List[str]) -> str:
                    try:
                        qdrant_end = os.getenv('QDRANT_URL')
                        qdrant_api_key = os.getenv('QDRANT_API_KEY')
                        openai_api_key = os.getenv('OPENAI_API_KEY')

                        embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)
                        qdrant_client = QdrantClient(url=qdrant_end, api_key=qdrant_api_key)
                        qdrant = Qdrant(client=qdrant_client, collection_name="policy-agent", embeddings=embeddings_model)
                        retriever = qdrant.as_retriever(search_kwargs={"k": 3})
                        responses = []
                        for q in query:
                            response = retriever.invoke(q)
                            responses.append((response))

                        return responses

                    except Exception as e:
                        logger.error("Error during report tool execution.")
                        raise CustomException(e, sys)

            logger.info("Report tool created successfully.")
            return ReportTool()
        except Exception as e:
            logger.error("Error during report tool creation.")
            raise CustomException(e, sys)

    def create_agent(self, system_message: str, tools=None):
        try:
            tool_names = ", ".join([tool.name for tool in tools]) if tools else "None"
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful AI assistant, collaborating with other assistants."
                        " Use the provided tools to progress towards answering the question."
                        " If you are unable to fully answer, that's OK, another assistant with different tools "
                        " will help where you left off. Execute what you can to make progress."
                        " If you or any of the other assistants have the final answer or deliverable,"
                        " prefix your response with FINAL ANSWER so the team knows to stop."
                        " You have access to the following tools: {tool_names}.\n{system_message}",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            prompt = prompt.partial(system_message=system_message, tool_names=tool_names)
            if tools:
                agent = prompt | self.llm.bind_tools(tools)
            else:
                agent = prompt | self.llm
            logger.info(f"Agent created successfully with system message: {system_message}")
            return agent
        except Exception as e:
            logger.error("Error during agent creation.")
            raise CustomException(e, sys)

    def agent_node(self, state, agent, name):
        try:
            logger.info(f"Agent '{name}' is processing the state.")
            result = agent.invoke(state)
            logger.info(f"Result from agent '{name}': {result}")

            if isinstance(result, ToolMessage):
                logger.info(f"ToolMessage from agent '{name}': {result}")
            else:
                result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
                logger.info(f"AIMessage from agent '{name}': {result}")

            return {
                "messages": [result],
                "sender": name,
            }
        except Exception as e:
            logger.error(f"Error in agent node '{name}'.")
            raise CustomException(e, sys)

    def _setup_workflow(self):
        try:
            summary_agent = self.create_agent(
                system_message="Summarize the user query. You should include all important data like financial figures, dates, etc. Output format should be a python list of strings.",
            )
            summary_node = functools.partial(self.agent_node, agent=summary_agent, name="Summarizer")

            policy_agent = self.create_agent(
                system_message=(
                    "You have the following tasks"
                    "1.Create a single comprehensive question from the summary provided by summary agent which includes all financial, date, and project-related data"
                    "2.Prepend 'What are the compliance criteria, eligibility criteria, fees' at the beginning of the question."
                    "3.Use the question to call_tool."
                    "4.Your input to the tool has to be a list of python string."
                    "5.After receiving the docs from the tool, extract the information related to the input question into a single document having headers and sub-heading"
                    "Keep in view the following points:"
                    "1.Remember you have to craft the question from the text not related to the text."
                    "2.While drafting the document, keep in mind the input question."
                    "3.Do not finalize the answer or add 'FINAL ANSWER' in your response."
                    "4.Return 'continue' after creating the document to call the next agent properly"
                ),
                tools=[self.report_tool_instance]
            )
            policy_node = functools.partial(self.agent_node, agent=policy_agent, name="policy_generator")

            financial_agent = self.create_agent(
                system_message=(
                    "You have the following tasks"
                    "1.Create a single comprehensive question from the summary provided by summary agent which includes all financial, date, and project-related data"
                    "2.Prepend ' What are the financing options, subsidies, grants, and incentives available' at the beginning of the question."
                    "3.Use the question to call_tool."
                    "4.Your input to the tool has to be a list of python string."
                    "5.After receiving the docs from the tool, extract the information related to the input question into a single document having headers and sub-heading"
                    "Keep in view the following points:"
                    "1.Remember you have to craft the question from the text not related to the text."
                    "2.Do not finalize the answer or add 'FINAL ANSWER' in your response."
                    "3.While drafting the document, keep in mind the input question."
                ),
                tools=[self.report_tool_instance]
            )
            finance_node = functools.partial(self.agent_node, agent=financial_agent, name="finance_generator")

            report_agent = self.create_agent(
                system_message=(
                    "You have the following tasks"
                    "1.Collect the answers generated by policy_agenst and financial agent"
                    "2.Read the answers by both agensts"
                    "3.Create a commulative report from these two agents responses"
                    "4.Be detail oriented, Genuine and Act as an expert report generator"
                    "5.prefix your response with FINAL ANSWER so the team knows to stop."
                ),
            )
            report_node = functools.partial(self.agent_node, agent=report_agent, name="report_generator")

            tools = [self.report_tool_instance]
            tool_node = ToolNode(tools)

            def router(state) -> Literal["call_tool", "__end__", "continue"]:
                messages = state["messages"]
                last_message = messages[-1]
                logger.info(f"Router checking the last message from sender '{state['sender']}': {last_message}")

                if last_message.tool_calls:
                    logger.info("Router directing to call_tool.")
                    return "call_tool"
                if "FINAL ANSWER" in last_message.content:
                    logger.info("Router directing to __end__.")
                    return "__end__"
                logger.info("Router directing to continue.")
                return "continue"

            self.workflow.add_node("Summarizer", summary_node)
            self.workflow.add_node("policy_generator", policy_node)
            self.workflow.add_node("finance_generator", finance_node)
            self.workflow.add_node("report_generator", report_node)
            self.workflow.add_node("call_tool", tool_node)

            self.workflow.add_conditional_edges(
                "Summarizer",
                router,
                {"continue": "policy_generator", "call_tool": "call_tool", "__end__": END},
            )
            self.workflow.add_conditional_edges(
                "policy_generator",
                router,
                {"continue": "finance_generator", "call_tool": "call_tool", "__end__": END},
            )
            self.workflow.add_conditional_edges(
                "finance_generator",
                router,
                {"continue": "report_generator", "call_tool": "call_tool", "__end__": END},
            )
            self.workflow.add_conditional_edges(
                "report_generator",
                router,
                {"continue": "__end__", "call_tool": "__end__", "__end__": END},
            )
            self.workflow.add_conditional_edges(
                "call_tool",
                lambda x: x["sender"],
                {
                    "Summarizer": "Summarizer",
                    "policy_generator": "policy_generator",
                    "finance_generator": "finance_generator"
                },
            )
            self.workflow.set_entry_point("Summarizer")
            logger.info("Workflow setup completed successfully.")
        except Exception as e:
            logger.error("Error during workflow setup.")
            raise CustomException(e, sys)

    def run(self, initial_message: str) -> str:
        try:
            final_state = self.graph.invoke(
                {
                    "messages": [
                        HumanMessage(content=initial_message)
                    ]
                },
                {"recursion_limit": 150}
            )
            final_response = final_state["messages"][-1]
            logger.info("Workflow run completed successfully.")
            logger.info(f"Final response content: {final_response.content}")
            return final_response.content 
        except Exception as e:
            logger.error("Error during workflow run.")
            raise CustomException(e, sys)