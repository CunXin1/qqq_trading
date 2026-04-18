"""
FastAPI application factory — creates and configures the web app.
FastAPI 应用工厂 — 创建并配置 Web 应用。

GET Routes / GET 路由:
    /         — Dashboard: current signal, recent alerts, task status.
                仪表盘：当前信号、最近告警、任务状态。
    /signal   — Detailed prediction view for the latest trading day.
                最新交易日的详细预测视图。
    /history  — Historical signal list with hit/miss classification and metrics.
                历史信号列表，包含命中/未命中分类及评估指标。
    /data     — Data file freshness and row-count status.
                数据文件新鲜度及行数状态。
    /model    — Model metadata, feature list, and evaluation metrics (AUC, AP, Brier).
                模型元数据、特征列表及评估指标（AUC、AP、Brier）。

POST Routes / POST 路由:
    /actions/fetch   — Trigger background data fetch + merge (redirects to /).
                       触发后台数据抓取与合并（重定向至 /）。
    /actions/predict — Trigger background prediction refresh (redirects to /signal).
                       触发后台预测刷新（重定向至 /signal）。
"""
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
