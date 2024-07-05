import logging
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.qdrant import Qdrant

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
            ValueError: If the URL is not a string.
            RuntimeError: If there is an error loading the URL or no data is loaded.
        """
        try:
            logging.info(f"Loading from URL: {url}")
            if not isinstance(url, str):
                raise ValueError("URL must be a string")
            loader = PDFPlumberLoader(url)
            data = loader.load()
            if data is None:
                raise RuntimeError(f"No data loaded from URL {url}")
            return data
        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise
        except Exception as e:
            logging.error(f"Error loading URL {url}: {str(e)}")
            raise RuntimeError(f"Error loading URL {url}: {str(e)}")

    def load_from_file(self, file_path):
        """
        Load PDF documents from a file path.

        Args:
            file_path (str): The file path of the PDF document.

        Returns:
            data: Loaded data from the PDF.

        Raises:
            ValueError: If the file path is not a string.
            RuntimeError: If there is an error loading the file or no data is loaded.
        """
        try:
            logging.info(f"Loading from file: {file_path}")
            if not isinstance(file_path, str):
                raise ValueError("File path must be a string")
            loader = PyPDFLoader(file_path)
            data = loader.load()
            if data is None:
                raise RuntimeError(f"No data loaded from file {file_path}")
            return data
        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {str(e)}")
            raise RuntimeError(f"Error loading file {file_path}: {str(e)}")

    def split_and_store(self, data):
        """
        Split and store the documents.

        Args:
            data: The data to split and store.

        Raises:
            ValueError: If there is no data to split and store.
        """
        if not data:
            raise ValueError("No data to split and store")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=250
        )
        docs = text_splitter.split_documents(data)
        self.all_docs.extend(docs)
        logging.info("Documents split and stored successfully.")

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

            logging.info("RAG system created successfully.")
            return "RAG system created successfully with the given policy documents."

        except Exception as e:
            logging.error(f"Error creating RAG system: {str(e)}")
            return str(e)
