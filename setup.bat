@echo off
echo ================================================
echo  Navigation Assistance System — Setup
echo ================================================

python -m venv venv
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Preparing directories...
python prepare_dataset.py

echo.
echo ================================================
echo  Setup complete! Choose how to start:
echo.
echo  1. Web App (full UI):
echo     python app.py
echo     Open: http://127.0.0.1:5000
echo.
echo  2. Quick Demo (no training needed):
echo     python demo.py
echo.
echo  3. Synthetic Dataset + Train:
echo     python download_dataset.py --dataset synthetic
echo     python train.py
echo ================================================
pause
