@echo off
echo Starting NCERT Doubt Solver...

:: Go to project directory
cd /d C:\Users\WG0088-14\ncert-doubt-solver

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Set Python path
set PYTHONPATH=C:\Users\WG0088-14\ncert-doubt-solver

:: Run the server
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

pauseS