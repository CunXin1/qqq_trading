"""FastAPI application factory."""
from pathlib import Path
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from server import services

_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(_DIR / "templates"))


def create_app() -> FastAPI:
    app = FastAPI(title="QQQ Trading Dashboard")
    app.mount("/static", StaticFiles(directory=str(_DIR / "static")), name="static")

    @app.get("/")
    async def dashboard(request: Request):
        data = services.get_dashboard()
        return templates.TemplateResponse("dashboard.html", {"request": request, **data})

    @app.get("/signal")
    async def signal(request: Request):
        data = services.get_signal_detail()
        return templates.TemplateResponse("signal.html", {"request": request, **data})

    @app.get("/history")
    async def history(request: Request, start: str = "", threshold: float = 0.3):
        data = services.get_history(start=start, threshold=threshold)
        return templates.TemplateResponse("history.html", {"request": request, **data})

    @app.get("/data")
    async def data_status(request: Request):
        data = services.get_data_status()
        return templates.TemplateResponse("data_status.html", {"request": request, **data})

    @app.get("/model")
    async def model_info(request: Request):
        data = services.get_model_info()
        return templates.TemplateResponse("model_info.html", {"request": request, **data})

    @app.post("/actions/fetch")
    async def action_fetch(background_tasks: BackgroundTasks):
        background_tasks.add_task(services.trigger_fetch)
        return RedirectResponse("/", status_code=303)

    @app.post("/actions/predict")
    async def action_predict(background_tasks: BackgroundTasks):
        background_tasks.add_task(services.trigger_predict)
        return RedirectResponse("/signal", status_code=303)

    return app


app = create_app()
