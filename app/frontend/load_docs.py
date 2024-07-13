import sys
import os

# Ensure the backend module is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from custom_logger import logger
from custom_exceptions import CustomException
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.qdrant import Qdrant

class PDFProcessor:
    """
    A class to process PDF files and create a retrieval-augmented generation (RAG) system.
    
    Attributes:
        openai_api_key (str): API key for OpenAI.
        qdrant_url (str): URL for Qdrant vector store.
        qdrant_api_key (str): API key for Qdrant vector store.
    """

    def __init__(self, openai_api_key, qdrant_url, qdrant_api_key):
        """
        Initialize the PDFProcessor with necessary API keys and URLs.

        Args:
            openai_api_key (str): API key for OpenAI.
            qdrant_url (str): URL for Qdrant vector store.
            qdrant_api_key (str): API key for Qdrant vector store.
        """
        self.openai_api_key = openai_api_key
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.all_docs = []

    def load_from_url(self, url):
        """
        Load PDF documents from a URL.

        Args:
            url (str): The URL of the PDF document.

        Returns:
            data: Loaded data from the PDF.

        Raises:
            CustomException: If there is an error loading the URL.
        """
        try:
            logger.info(f"Loading from URL: {url}")
            if not isinstance(url, str):
                raise CustomException("URL must be a string", sys)
            loader = PDFPlumberLoader(url)
            data = loader.load()
            if data is None:
                raise CustomException(f"No data loaded from URL {url}", sys)
            return data
        except Exception as e:
            logger.error(f"Error loading URL {url}: {str(e)}")
            raise CustomException(f"Error loading URL {url}: {str(e)}", sys)

    def load_from_file(self, file_path):
        """
        Load PDF documents from a file path.

        Args:
            file_path (str): The file path of the PDF document.

        Returns:
            data: Loaded data from the PDF.

        Raises:
            CustomException: If there is an error loading the file.
        """
        try:
            logger.info(f"Loading from file: {file_path}")
            if not isinstance(file_path, str):
                raise CustomException("File path must be a string", sys)
            loader = PyPDFLoader(file_path)
            data = loader.load()
            if data is None:
                raise CustomException(f"No data loaded from file {file_path}", sys)
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise CustomException(f"Error loading file {file_path}: {str(e)}", sys)

    def split_and_store(self, data):
        """
        Split and store the documents.

        Args:
            data: The data to split and store.

        Raises:
            CustomException: If there is no data to split and store.
        """
        try:
            if not data:
                raise CustomException("No data to split and store", sys)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=250
            )
            docs = text_splitter.split_documents(data)
            self.all_docs.extend(docs)
            logger.info("Documents split and stored successfully.")
        except Exception as e:
            logger.error(f"Error splitting and storing documents: {str(e)}")
            raise CustomException(f"Error splitting and storing documents: {str(e)}", sys)

    def create_rag_system(self):
        """
        Create a retrieval-augmented generation (RAG) system.

        Returns:
            str: Success message if the RAG system is created successfully, else the error message.
        """
        try:
            embeddings_model = OpenAIEmbeddings(
                model='text-embedding-ada-002',
                openai_api_key=self.openai_api_key
            )

            Qdrant.from_documents(
                self.all_docs,
                embeddings_model,
                url=self.qdrant_url,
                prefer_grpc=True,
                api_key=self.qdrant_api_key,
                collection_name="policy-agent",
            )

            logger.info("RAG system created successfully.")
            return "RAG system created successfully with the given policy documents."

        except Exception as e:
            logger.error(f"Error creating RAG system: {str(e)}")
            raise CustomException(f"Error creating RAG system: {str(e)}", sys)
