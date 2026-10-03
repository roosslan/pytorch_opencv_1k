# Подготовка окружения rec_pytorch: пакеты в venv, пароль в .env, ClickHouse в Docker,
# схема базы, самопроверка и тесты. Запуск из директории проекта:
#   powershell -ExecutionPolicy Bypass -File .\setup_dev.ps1
# Скрипт можно запускать повторно: готовые шаги просто подтверждаются.

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'

function Step([string]$title) { Write-Host "`n=== $title" -ForegroundColor Cyan }

function Run([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Ошибка (код ${LASTEXITCODE}): $exe $($arguments -join ' ')" -ForegroundColor Red
        exit 1
    }
}

Step 'Виртуальное окружение'
if (-not (Test-Path $py)) {
    Run 'python' @('-m', 'venv', 'venv')
}
Run $py @('--version')

Step 'Пакеты Python (torch с CUDA уже стоит и не переустанавливается)'
Run $py @('-m', 'pip', 'install', '-r', 'requirements.txt', '-r', 'requirements-dev.txt')
Run $py @('-c', "import torch, ultralytics, cv2, numpy, clickhouse_connect; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())")

Step 'Пароль ClickHouse (.env)'
$envFile = Join-Path $PSScriptRoot '.env'
if (Test-Path $envFile) {
    Write-Host '.env уже есть, оставляю как есть'
} else {
    $bytes = New-Object byte[] 18
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $password = [Convert]::ToBase64String($bytes).Replace('+', 'x').Replace('/', 'y').TrimEnd('=')
    $text = "# Пароль ClickHouse для docker-compose и rec_pytorch. Файл .env в git не добавляйте.`n" +
            "CLICKHOUSE_PASSWORD=$password`nREC_CLICKHOUSE_PASSWORD=$password`n"
    [IO.File]::WriteAllText($envFile, $text, (New-Object Text.UTF8Encoding $false))
    Write-Host 'Создан .env со случайным паролем'
}

Step 'ClickHouse в Docker'
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Docker не отвечает: запустите Docker Desktop и повторите скрипт' -ForegroundColor Red
    exit 1
}
Run 'docker' @('compose', 'up', '-d')
Write-Host 'Жду, пока ClickHouse начнёт отвечать...'
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $reply = Invoke-WebRequest -Uri 'http://127.0.0.1:8123/ping' -UseBasicParsing -TimeoutSec 2
        if ($reply.Content -match 'Ok') { $ready = $true; break }
    } catch { }
    Start-Sleep -Seconds 2
}
if (-not $ready) {
    Write-Host 'ClickHouse не ответил за 2 минуты. Журнал: docker logs rec_clickhouse' -ForegroundColor Red
    exit 1
}
Write-Host 'ClickHouse готов'

Step 'Схема базы'
Run $py @('main.py', '--init-db')

Step 'Самопроверка модели'
Run $py @('main.py', '--self-test')

Step 'Проверка кода и тесты'
Run $py @('-m', 'ruff', 'check', '.')
Run $py @('-m', 'pytest', '-q')

Write-Host "`nВсё готово. Запуск: .\venv\Scripts\python.exe main.py" -ForegroundColor Green
