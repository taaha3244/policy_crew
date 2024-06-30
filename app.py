from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import CrewManager, OpenAIResponseModel
from custom_logger import get_logger

app = FastAPI()

logger = get_logger(__name__)

class QueryModel(BaseModel):
    query: str

@app.post("/process_query/")
async def process_query(query: QueryModel):
    try:
        manager = CrewManager(query.query)
        openai_response = manager.get_openai_response()
        logger.info(f"OpenAI response: {openai_response}")
        result = manager.start_crew(openai_response.is_generic)
        return {"result": result}
    except Exception as e:
        logger.exception("Error processing the query")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
