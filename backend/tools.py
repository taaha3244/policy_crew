import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.qdrant import Qdrant
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from crewai_tools import BaseTool
from typing import List
from backend.custom_logger import get_logger
from backend.custom_exceptions import RetrievalError, DataProcessingError

# Load environment variables
load_dotenv()

# Initialize the logger
logger = get_logger(__name__)

class RAGTool:
    """
    A class to handle the Retrieval-Augmented Generation (RAG) process.

    Attributes:
        query (str): The query to process using the RAG system.
    """

    def __init__(self, query: str):
        """
        Initialize the RAGTool with the given query.

        Args:
            query (str): The query to process.
        """
        self.query = query

    def qa_from_RAG(self) -> str:
        """
        Process the query using the RAG system and return the result.

        Returns:
            str: The result of processing the query.

        Raises:
            RetrievalError: If there is an error retrieving or processing the query.
        """
        try:
            logger.info("Initializing RAG tool with query: %s", self.query)

            # Setup
            qdrant_url = os.getenv('QDRANT_URL')
            qdrant_api_key = os.getenv('QDRANT_API_KEY')
            openai_api_key = os.getenv('OPENAI_API_KEY')

            if not qdrant_url or not qdrant_api_key or not openai_api_key:
                raise RetrievalError("Missing environment variables for Qdrant or OpenAI")

            embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            qdrant = Qdrant(client=qdrant_client, collection_name="policy-agent", embeddings=embeddings_model)
            retriever = qdrant.as_retriever(search_kwargs={"k": 5})
            prompt = hub.pull('pwoc517/more-crafted-rag-prompt')
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_api_key)

            def format_docs(docs):
                """
                Format the documents by combining page content with its metadata.

                Args:
                    docs (list): List of documents to format.

                Returns:
                    str: Formatted documents as a string.
                """
                formatted_docs = []
                for doc in docs:
                    metadata_str = ', '.join(f"{key}: {value}" for key, value in doc.metadata.items())
                    doc_str = f"{doc.page_content}\nMetadata: {metadata_str}"
                    formatted_docs.append(doc_str)
                return "\n\n".join(formatted_docs)

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            result = rag_chain.invoke(self.query)
            logger.info("Query processed successfully: %s", self.query)
            return result
        except Exception as e:
            logger.exception("Error processing the query")
            raise RetrievalError(f"Error processing the query: {e}")


class ReportTool(BaseTool):
    """
    A tool to retrieve relevant documents from the vector database using user queries.
    """
    name: str = "Report Tool"
    description: str = "Tool to retrieve relevant documents from the vector database using a list of user queries and return a response."

    def _run(self, queries: List[str]) -> List[str]:
        """
        Run the tool with the provided queries and return the results.

        Args:
            queries (List[str]): The list of queries to process.

        Returns:
            List[str]: The list of retrieved documents.

        Raises:
            DataProcessingError: If there is an error processing the queries.
        """
        try:
            logger.info("Running report tool with queries: %s", queries)

            # Setup
            qdrant_url = os.getenv('QDRANT_URL')
            qdrant_api_key = os.getenv('QDRANT_API_KEY')
            openai_api_key = os.getenv('OPENAI_API_KEY')

            if not qdrant_url or not qdrant_api_key or not openai_api_key:
                raise RetrievalError("Missing environment variables for Qdrant or OpenAI")

            embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            qdrant = Qdrant(client=qdrant_client, collection_name="policy-agent", embeddings=embeddings_model)

            responses = []
            for query in queries:
                # Embed the input query for vector search
                query_result = embeddings_model.embed_query(query)

                # Perform vector search in the "policy-agent" collection
                response = qdrant_client.search(
                    collection_name="policy-agent",
                    query_vector=query_result,
                    limit=2  # Retrieve top 2 closest vectors
                )

                responses.append(response)

            logger.info("Queries processed successfully: %s", queries)
            return responses
        except Exception as e:
            logger.exception("Error processing the queries")
            raise DataProcessingError(f"Error processing the queries: {e}")
