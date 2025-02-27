"""
Main web application service. Serves the static frontend.
"""
from pathlib import Path
import modal

from modal_tts import TTSService  # makes modal deploy also deploy TTSService
from common import app

static_path = Path(__file__).with_name("frontend").resolve()
print("static_path: ", static_path)

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi",
    "jinja2"
).add_local_dir(static_path, remote_path="/assets")

@app.function(
    container_idle_timeout=600,
    timeout=600,
    allow_concurrent_inputs=100,
    image=image,
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates
    from fastapi.requests import Request

    web_app = FastAPI()

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files first
    web_app.mount("/static", StaticFiles(directory="/assets"), name="static")
    
    templates = Jinja2Templates(directory="/assets")

    @web_app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    return web_app