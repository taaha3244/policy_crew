import sys
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.backend.main import CrewManager, LangraphManager  # Import LangraphManager
from custom_logger import logger
from app.backend.database import redis_client  # Import redis_client from database.py
from custom_exceptions import CustomException

app = FastAPI()

class QueryModel(BaseModel):
    query: str

class QuestionResponse(BaseModel):
    question: str
    response: str
    agent: str

@app.post("/process_query/")
async def process_query(query: QueryModel):
    """Endpoint to process a query using CrewManager and save the response to Redis.

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
        
        agent_name = "Crew AI RAG" if openai_response.is_generic else "Crew AI AI agent"
        
        question_response = QuestionResponse(
            question=query.query, 
            response=result,
            agent=agent_name
        )
        
        # Attempt to save the question and response to Redis
        try:
            redis_key = f"question:{query.query}"
            redis_value = question_response.json()
            redis_client.set(redis_key, redis_value)
            logger.info("Successfully saved data to Redis")
        except redis.ConnectionError as redis_error:
            logger.exception("Redis error occurred while saving the response")
        
        return {"result": result}
    
    except CustomException as ce:
        logger.error(f"CustomException: {str(ce)}")
        raise HTTPException(status_code=500, detail=str(ce))
    except Exception as e:
        logger.exception("Unexpected error occurred while processing the query")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/process_query_langraph/")
async def process_query_langraph(query: QueryModel):
    """Endpoint to process a query using LangraphManager and save the response to Redis.

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
        
        agent_name = "Langraph Graph RAG" if langraph_manager.crew_manager.get_openai_response().is_generic else "Langraph AI agent"
        
        question_response = QuestionResponse(
            question=query.query, 
            response=result,
            agent=agent_name
        )
        
        # Attempt to save the question and response to Redis
        try:
            redis_key = f"question:{query.query}"
            redis_value = question_response.json()
            redis_client.set(redis_key, redis_value)
            logger.info("Successfully saved data to Redis")
        except redis.ConnectionError as redis_error:
            logger.exception("Redis error occurred while saving the response")
        
        return {"result": result}
    
    except CustomException as ce:
        logger.error(f"CustomException: {str(ce)}")
        raise HTTPException(status_code=500, detail=str(ce))
    except Exception as e:
        logger.exception("Unexpected error occurred while processing the query")
        raise HTTPException(status_code=500, detail="Internal server error")
