"""Entry point: python -m qqq_trading.server"""
import argparse
import uvicorn
from qqq_trading.server.app import create_app

def main():
    parser = argparse.ArgumentParser(description="QQQ Trading Dashboard")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
