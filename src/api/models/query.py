from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the agent")
    user_id: str = Field(..., description="Unique identifier for the user")

class QueryResponse(BaseModel):
    response: str = Field(..., description="The agent's response to the question")