@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Running Multi-Robot Fragment Planner...
python src\main.py --map data\map.yaml
pause
