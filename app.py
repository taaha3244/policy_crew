from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import CrewManager, OpenAIResponseModel
from custom_logger import get_logger
from database import get_database
from pymongo.errors import PyMongoError
from bson.objectid import ObjectId

app = FastAPI()

logger = get_logger(__name__)

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
    try:
        manager = CrewManager(query.query)
        openai_response = manager.get_openai_response()
        logger.info(f"OpenAI response: {openai_response}")
        result = manager.start_crew(openai_response.is_generic)

        # Save the question and response to MongoDB
        question_response = QuestionResponse(question=query.query, response=result)
        save_result = questions_collection.insert_one(question_response.model_dump())
        if not save_result.acknowledged:
            raise HTTPException(status_code=500, detail="Failed to save data")

        return {"result": result}
    except Exception as e:
        logger.exception("Error processing the query")
        raise HTTPException(status_code=500, detail=str(e))


