import os
import sys

# Ensure the project root is at the top of sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

import giskard
import pandas as pd
from giskard.rag import evaluate, KnowledgeBase, generate_testset, QATestset
from giskard.llm.client.openai import OpenAIClient
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.backend.utils import get_hyperparameters_from_file
from custom_logger import logger
from custom_exceptions import CustomException
from app.backend.tools import RAGTool

# Load configuration and environment variables
config = get_hyperparameters_from_file()
load_dotenv()

# Set up environment variables
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Set up Giskard with OpenAI
giskard.llm.set_llm_api("openai")
oc = OpenAIClient(model=config['LLM_NAME'])
giskard.llm.set_default_client(oc)

class GiskardEvals:
    def __init__(self):
        self.file_path = project_root
        self.pdf_paths = self.load_paths()
        self.text_chunks, self.knowledge_base = self.load_and_split_docs()
        self.testset_path = os.path.join(self.file_path, "test-set.jsonl")
        self.testset = self.load_or_generate_testset()

    def load_paths(self):
        """
        Load paths of PDF files from the specified directory.
        """
        pdf_paths = []
        try:
            for root, _, files in os.walk(self.file_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        full_path = os.path.join(root, file)
                        pdf_paths.append(full_path)
            if pdf_paths:
                logger.info("PDF paths loaded successfully.")
            else:
                logger.warning("No PDF files found in the specified directory.")
        except Exception as e:
            logger.error(f"Error loading paths: {e}")
            raise CustomException(e, sys)
        return pdf_paths
    
    def load_and_split_docs(self):
        """
        Load and split documents into text chunks.
        """
        documents = []
        try:
            for file_path in self.pdf_paths:
                loader = PyPDFLoader(file_path)
                loaded_documents = loader.load()
                documents.extend(loaded_documents)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            text_chunks = text_splitter.split_documents(documents)

            df = pd.DataFrame([d.page_content for d in documents], columns=["text"])
            knowledge_base = KnowledgeBase(df)

            logger.info("Documents loaded and split successfully.")
            return text_chunks, knowledge_base
        except Exception as e:
            logger.error(f"Error loading and splitting documents: {e}")
            raise CustomException(e, sys)
    
    def load_or_generate_testset(self):
        """
        Load test set from file if available, otherwise generate a new one.
        """
        if os.path.exists(self.testset_path):
            try:
                testset = QATestset.load(self.testset_path)
                logger.info(f"Test set loaded from {self.testset_path}.")
            except Exception as e:
                logger.error(f"Error loading test set: {e}")
                raise CustomException(e, sys)
        else:
            try:
                testset = generate_testset(
                    knowledge_base=self.knowledge_base,
                    num_questions=30,
                    agent_description="A chatbot answering questions about different policies related to projects",
                )
                testset.save(self.testset_path)
                logger.info(f"Test set generated and saved to {self.testset_path}.")
            except Exception as e:
                logger.error(f"Error generating test set: {e}")
                raise CustomException(e, sys)
        return testset

    def load_rag_for_eval(self, question, history=None):
        """
        Load RAG tool for evaluation.
        """
        try:
            rag_tool = RAGTool(question)
            return rag_tool.qa_from_RAG()
        except Exception as e:
            logger.error(f"Error loading RAG tool: {e}")
            raise CustomException(e, sys)
    
    def giskard_report(self):
        """
        Generate a Giskard report for the evaluation.
        """
        try:
            report = evaluate(self.load_rag_for_eval, testset=self.testset, knowledge_base=self.knowledge_base)
            report_path = os.path.join(self.file_path, "giskard-report.html")
            report.to_html(report_path)
            logger.info("Giskard report generated successfully.")
            return report
        except Exception as e:
            logger.error(f"Error generating Giskard report: {e}")
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    try:
        giskard_eval = GiskardEvals()
        giskard_eval.giskard_report()
    except CustomException as e:
        logger.error(f"An error occurred during evaluation: {e}")
