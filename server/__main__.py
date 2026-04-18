"""
Entry point for running the server: ``python -m server``.
服务器启动入口：通过 ``python -m server`` 运行。

Launches a uvicorn ASGI server hosting the QQQ Trading Dashboard.
Default listen address is 0.0.0.0:8000; configurable via --host / --port.
启动 uvicorn ASGI 服务器来托管 QQQ 交易仪表盘。
默认监听地址为 0.0.0.0:8000，可通过 --host / --port 参数自定义。
"""
import argparse
import uvicorn
from server.app import create_app

def main():
    parser = argparse.ArgumentParser(description="QQQ Trading Dashboard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
