# Activate venv (if present) and run the Streamlit app
$venv = Join-Path -Path $PSScriptRoot -ChildPath ".venv\Scripts\python.exe"
if (Test-Path $venv) {
    & $venv -m streamlit run "app/streamlit_app.py" --server.port 8501
} else {
    Write-Host ".venv not found. Create virtual environment first: python -m venv .venv" -ForegroundColor Yellow
}
