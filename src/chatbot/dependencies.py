"""Depends() providers, e.g. the loaded predictor."""

from typing import Annotated

from fastapi import Depends, Request

from src.chatbot.predictor import Predictor


def get_predictor(request: Request) -> Predictor:
    """The predictor loaded once at startup (see the lifespan in main.py)."""
    return request.app.state.predictor


PredictorDep = Annotated[Predictor, Depends(get_predictor)]
