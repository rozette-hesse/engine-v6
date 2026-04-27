"""
Development server launcher.

Run: python serve.py

For production, use uvicorn directly:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

import os
import sys

# Ensure the project root is always on the path, regardless of working directory
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import socket
import uvicorn
from core.config import settings


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


if __name__ == "__main__":
    ip = _lan_ip()
    print("\n  inBalance is running")
    print(f"  Local:   http://localhost:{settings.port}")
    print(f"  Network: http://{ip}:{settings.port}  ← share this with others on your WiFi\n")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
