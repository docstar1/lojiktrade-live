@echo off
REM ===========================================================================
REM  LOJIKTRADE-LIVE | local launcher
REM  Pulls latest, runs scanner_worker.py to refresh ledger, then opens dashboard.
REM ===========================================================================
setlocal
cd /d "%~dp0"

REM --- Auto-catchup: pull latest (ignore offline errors) ---
git pull --rebase --autostash >nul 2>&1

REM --- Refresh ledger to latest prices ---
python scanner_worker.py

REM --- Open dashboard ---
start "" index.html

endlocal