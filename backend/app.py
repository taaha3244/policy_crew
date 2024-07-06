import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.main import CrewManager, LangraphManager  # Import LangraphManager
from backend.custom_logger import logger
from backend.database import get_database
from pymongo.errors import PyMongoError
from backend.custom_exceptions import CustomException

app = FastAPI()

# Get the database
db = get_database()
questions_collection = db["questions"]

class QueryModel(BaseModel):
    query: str

class QuestionResponse(BaseModel):
    question: str
    response: str

@app.post("/process_query/")
async def process_query(query: QueryModel):
    """Endpoint to process a query using CrewManager and save the response to MongoDB.

    Args:
        query (QueryModel): The query model containing the user's query.

    Returns:
        dict: A dictionary containing the result of the processed query.

    Raises:
        HTTPException: If there is an error processing the query.
    """
    try:
        manager = CrewManager(query.query)
        openai_response = manager.get_openai_response()
        logger.info(f"OpenAI response: {openai_response}")

        result = manager.start_crew(openai_response.is_generic)
        question_response = QuestionResponse(question=query.query, response=result)
        
        # Attempt to save the question and response to MongoDB
        try:
            save_result = questions_collection.insert_one(question_response.dict())
            if not save_result.acknowledged:
                logger.error("Failed to save data to MongoDB")
        except PyMongoError as db_error:
            logger.exception("MongoDB error occurred while saving the response")
        
        return {"result": result}
    
    except CustomException as ce:
        logger.error(f"CustomException: {str(ce)}")
        raise HTTPException(status_code=500, detail=str(ce))
    except Exception as e:
        logger.exception("Unexpected error occurred while processing the query")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/process_query_langraph/")
async def process_query_langraph(query: QueryModel):
    """Endpoint to process a query using LangraphManager and save the response to MongoDB.

    Args:
        query (QueryModel): The query model containing the user's query.

    Returns:
        dict: A dictionary containing the result of the processed query.

    Raises:
        HTTPException: If there is an error processing the query.
    """
    try:
        langraph_manager = LangraphManager(query.query)
        result = langraph_manager.run_workflow()
        if result is None:
            raise ValueError("Langraph workflow returned None")
        
        question_response = QuestionResponse(question=query.query, response=result)
        
        # Attempt to save the question and response to MongoDB
        try:
            save_result = questions_collection.insert_one(question_response.dict())
            if not save_result.acknowledged:
                logger.error("Failed to save data to MongoDB")
        except PyMongoError as db_error:
            logger.exception("MongoDB error occurred while saving the response")
        
        return {"result": result}
    
    except CustomException as ce:
        logger.error(f"CustomException: {str(ce)}")
        raise HTTPException(status_code=500, detail=str(ce))
    except Exception as e:
        logger.exception("Unexpected error occurred while processing the query")
        raise HTTPException(status_code=500, detail="Internal server error")
