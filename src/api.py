from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
# Assuming DocumentResearchAgent is in src.agent
# Adjust the import path if necessary
from agent import DocumentResearchAgent

app = FastAPI()

class ResearchRequest(BaseModel):
    query: str
    filenames: Optional[List[str]] = None

@app.post("/research")
async def research(request: ResearchRequest):
    try:
        agent = DocumentResearchAgent()
        if request.filenames:
            response = agent.run(request.query, request.filenames)
        else:
            response = agent.run(request.query)
        return {"response": response}
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"Error during research: {e}")
        # Return a generic error response
        raise HTTPException(status_code=500, detail="An error occurred during the research process.")

# To run this API, you would use a command like:
# uvicorn src.api:app --reload
# (assuming uvicorn is installed and you are in the project root)
