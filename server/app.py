"""
server/app.py — Multi-mode deployment entry point.
"""

from app import app
import uvicorn

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    main()
