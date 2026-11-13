@echo off
cd /d "%~dp0.."
set IMAGE_NAME=zac2026-lifemate:v1
set TAR_NAME=zac2026_LifeMate.tar.gz

echo === [1/3] Building %IMAGE_NAME% ===
docker build -f submission/Dockerfile -t %IMAGE_NAME% .
if errorlevel 1 exit /b 1

echo === [2/3] Exporting %TAR_NAME% ===
docker save %IMAGE_NAME% | gzip > %TAR_NAME%
if errorlevel 1 exit /b 1

echo === [3/3] MD5 ===
powershell -Command "(Get-FileHash -Algorithm MD5 %TAR_NAME%).Hash.ToLower()" > %TAR_NAME%.md5
type %TAR_NAME%.md5
