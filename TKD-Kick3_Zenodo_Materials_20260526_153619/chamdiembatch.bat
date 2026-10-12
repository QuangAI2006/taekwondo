@echo off
rem Mo app batch: chon thu muc, kiem tra dong tac, cham tung video lan luot.
rem Lan dau chay: tu tao .venv va cai thu vien theo code\requirements-lock.txt (can Python 3.12, mat vai phut).
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" goto run

echo [Lan dau] Tao moi truong Python .venv ...
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Khong tim thay Python. Hay cai Python 3.12 tu https://www.python.org/downloads/ roi chay lai.
    pause
    exit /b 1
)
%PY% -m venv .venv || goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto fail
echo Dang cai thu vien (torch, mediapipe, opencv...) - co the mat vai phut ...
".venv\Scripts\python.exe" -m pip install -r "code\requirements-lock.txt" || goto fail
echo Cai dat xong.

:run
start "" ".venv\Scripts\pythonw.exe" "code\scripts\score_batch_app.py"
exit /b 0

:fail
echo.
echo Cai dat that bai - xem loi o tren. Xoa thu muc .venv roi chay lai file nay.
pause
exit /b 1
