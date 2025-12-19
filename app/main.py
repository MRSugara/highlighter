from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes.web import router as web_router

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(web_router)
