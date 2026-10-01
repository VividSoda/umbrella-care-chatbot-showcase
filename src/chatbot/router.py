"""POST /api/v1/chatbot/ — HTTP only: parse, call service, return."""

from fastapi import APIRouter

from src.chatbot import service
from src.chatbot.dependencies import PredictorDep
from src.chatbot.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1/chatbot", tags=["chatbot"])


# Plain `def`, not `async def`: model inference is CPU-bound, so FastAPI runs
# it in the threadpool instead of blocking the event loop.
@router.post("/")
def chat(request: ChatRequest, predictor: PredictorDep) -> ChatResponse:
    return service.build_reply(request, predictor)
