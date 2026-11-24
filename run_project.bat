
@echo off

start "Flask Server" /D "C:\Users\NANI\OneDrive\Desktop\pavsre\Medical Chatbot final\Medical Chatbot" cmd /k "venv\Scripts\activate.bat && python app1.py"

timeout /t 10 /nobreak >nul

start "ngrok" /D "D:\ngrok-v3-stable-windows-amd64" cmd /k "ngrok.exe http 8000 --url https://unified-goldfish-alkaline.ngrok-free.dev"