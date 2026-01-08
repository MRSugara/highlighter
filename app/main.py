from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Load environment variables from .env for local development.
# In production, you can set env vars via your process manager instead.
try:
	from dotenv import load_dotenv  # type: ignore

	load_dotenv()
except Exception:
	pass

from app.routes.web import router as web_router

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(web_router)
