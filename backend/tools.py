import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.qdrant import Qdrant
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaindexOpenAIEmbeddings
from llama_index.llms.openai import OpenAI as LlamaindexOpenAI
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
import nest_asyncio
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)

from langchain import hub
from crewai_tools import BaseTool
from typing import List
from backend.custom_logger import logger
from backend.custom_exceptions import CustomException

# Load environment variables
load_dotenv()

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
            CustomException: If there is an error retrieving or processing the query.
        """
        try:
            logger.info("Initializing RAG tool with query: %s", self.query)

            # Setup
            qdrant_url = os.getenv('QDRANT_URL')
            qdrant_api_key = os.getenv('QDRANT_API_KEY')
            openai_api_key = os.getenv('OPENAI_API_KEY')
            jina_api_key=os.getenv('JINA_API_KEY')

            if not qdrant_url or not qdrant_api_key or not openai_api_key:
                raise CustomException("Missing environment variables for Qdrant , OpenAI or JINA", sys)

            embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            qdrant = Qdrant(client=qdrant_client, collection_name="policy-agent", embeddings=embeddings_model)
            retriever = qdrant.as_retriever(search_kwargs={"k": 20})
            prompt = hub.pull('pwoc517/more-crafted-rag-prompt')
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_api_key)
            compressor = JinaRerank(jina_api_key=jina_api_key,top_n=5)
            compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
            )

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
                {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            result = rag_chain.invoke(self.query)
            logger.info("Query processed successfully: %s", self.query)
            return result
        except Exception as e:
            logger.exception("Error processing the query")
            raise CustomException(f"Error processing the query: {e}", sys)


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
            CustomException: If there is an error processing the queries.
        """
        try:
            logger.info("Running report tool with queries: %s", queries)

            # Setup
            qdrant_url = os.getenv('QDRANT_URL')
            qdrant_api_key = os.getenv('QDRANT_API_KEY')
            openai_api_key = os.getenv('OPENAI_API_KEY')
            jina_api_key=os.getenv('JINA_API_KEY')

            if not qdrant_url or not qdrant_api_key or not openai_api_key:
                raise CustomException("Missing environment variables for Qdrant or OpenAI", sys)

            embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            qdrant = Qdrant(client=qdrant_client, collection_name="policy-agent", embeddings=embeddings_model)
            retriever = qdrant.as_retriever(search_kwargs={"k": 15})
            compressor = JinaRerank(jina_api_key=jina_api_key,top_n=3)
            compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
            )


            responses = []
            for query in queries:
                # Embed the input query for vector search
                query_result = compression_retriever.invoke(query)
                responses.append(query_result)

            logger.info("Queries processed successfully: %s", queries)
            return responses
        except Exception as e:
            logger.exception("Error processing the queries")
            raise CustomException(f"Error processing the queries: {e}", sys)


#Retrieval Class for project

class GraphRagTool:
    """ A class for creating Graph RAG tool for AI agents """

    def __init__(self, query):
        """
        Initialize the GraphRAG with the given query.

        Args:
            query (str): The query to process.
        """
        self.query = query
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.neo4j_url = os.getenv('NEO4J_URL')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.embed_model = LlamaindexOpenAIEmbeddings(model_name="text-embedding-3-small", api_key=self.openai_api_key)
        self.llm = LlamaindexOpenAI(model="gpt-3.5-turbo", temperature=0.0, api_key=self.openai_api_key)
        self.graph_store = Neo4jPropertyGraphStore(
            username="neo4j",
            password=self.neo4j_password,
            url=self.neo4j_url
        )

    def load_neo4j_graph(self):
        """
        Load from existing graph/vector store and process the query.

        Returns:
            str: The result of the query.

        Raises:
            CustomException: If there is an error retrieving or processing the query.
        """
        try:
            logger.info("Initializing Graph RAG Tool with query: %s", self.query)
            nest_asyncio.apply()
            # Load from existing graph/vector store
            index = PropertyGraphIndex.from_existing(
                property_graph_store=self.graph_store,
                embed_kg_nodes=True,
                llm=self.llm,
            )

            llm_synonym = LLMSynonymRetriever(
                index.property_graph_store,
                llm=self.llm,
                include_text=True,
            )
            vector_context = VectorContextRetriever(
                index.property_graph_store,
                embed_model=self.embed_model,
                include_text=True,
            )
            query_engine = index.as_query_engine(
                sub_retrievers=[llm_synonym, vector_context],
                include_text=True
            )

            response = query_engine.query(self.query)
            logger.info("Query processed successfully: %s", self.query)
            return response.response
        except Exception as e:
            logger.exception("Error processing the query")
            raise CustomException(f"Error processing the query: {e}", sys)

