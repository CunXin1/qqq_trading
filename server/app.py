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

    @app.on_event("startup")
    async def startup_refresh():
        """Auto-check data staleness on server startup and refresh via IBKR if stale.
        服务器启动时自动检查数据过期状态，如过期则通过 IBKR 刷新。"""
        try:
            from data.refresh import check_staleness, refresh_if_stale
            status = check_staleness()
            if status["stale"]:
                print(f"[startup] Data stale (last: {status['last_date']}, "
                      f"target: {status['target_date']}). Refreshing in background...")
                import asyncio
                asyncio.create_task(refresh_if_stale())
            else:
                print(f"[startup] Data fresh (last: {status['last_date']})")
        except Exception as e:
            print(f"[startup] Auto-refresh check failed: {e}")

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

    @app.get("/eval")
    async def eval_report(request: Request, task: str = "range_0dte",
                          model: str = "", threshold: float = 0.5,
                          start: str = "2026-01-01", thresh: float = 0.02,
                          miss_thresh: float = 3.0):
        data = services.get_eval_report(
            task=task, model=model or None, start=start, thresh=thresh,
            threshold=threshold, miss_thresh=miss_thresh,
        )
        return templates.TemplateResponse("eval.html", {"request": request, **data})

    @app.get("/eval/cross")
    async def cross_eval(request: Request):
        data = services.get_cross_eval()
        return templates.TemplateResponse("eval_cross.html", {"request": request, **data})

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
