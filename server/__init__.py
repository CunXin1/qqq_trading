"""
QQQ Trading Dashboard — FastAPI + Jinja2 Web Server.
QQQ 交易仪表盘 — FastAPI + Jinja2 Web 服务器。

This module provides a real-time web dashboard for displaying QQQ large-move
predictions, signal history, data freshness status, and model evaluation info.
本模块提供实时 Web 仪表盘，用于展示 QQQ 大幅波动预测、信号历史、数据状态和模型评估信息。

Submodules / 子模块:
    - app.py: FastAPI application factory with route definitions (5 GET + 2 POST).
      FastAPI 应用工厂，包含路由定义（5 个 GET + 2 个 POST）。
    - services.py: Business logic layer with file-mtime-based cache invalidation;
      wraps features/models/eval modules for web display.
      业务逻辑层，基于文件修改时间的缓存失效机制；封装特征/模型/评估模块供 Web 展示。
    - static/ & templates/: Jinja2 HTML templates and static assets (CSS/JS) for
      the frontend interface.
      Jinja2 HTML 模板和前端静态资源（CSS/JS）。
"""
from server.app import create_app
