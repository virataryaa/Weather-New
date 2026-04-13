@echo off
REM Coffee Weather — Daily Update + GitHub Push
REM Schedule this file in Windows Task Scheduler to run daily at 07:00 AM

SET REPO="C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Non Fundamental\Weather\Coffee"

cd /d %REPO%

echo [%date% %time%] Running daily update...
python daily_update.py

echo [%date% %time%] Pushing to GitHub...
git add data\*.parquet
git commit -m "Daily weather update %date%"
git push origin main

echo [%date% %time%] Done.
