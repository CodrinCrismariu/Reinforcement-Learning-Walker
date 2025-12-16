@echo off
call .\venv\Scripts\activate
tensorboard --logdir results --port 6006
pause
