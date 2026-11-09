@echo off
REM One-time setup on Windows:  setup.bat
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Done. Next time just run: .venv\Scripts\activate  and  set PYTHONPATH=src
