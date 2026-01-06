# AI Video Highlighter

Scaffolded project structure for an AI-powered video highlighter.

## Structure

- `venv/` — Python virtual environment (create via `python -m venv venv`)
- `app/` — Application source
  - `database/` — DB setup, models, migrations
  - `routes/` — Web routes / endpoints
  - `services/` — YouTube, transcript, AI logic, downloads
  - `templates/` — Jinja2 HTML templates
  - `static/` — CSS/JS assets
  - `schemas/` — Data models

## Database (MySQL)

This project uses MySQL via SQLAlchemy Async + aiomysql.

1. Create a database (example name: `ai_highlighter`):

```sql
CREATE DATABASE IF NOT EXISTS ai_highlighter CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. Set `DATABASE_URL` env var (or edit default in `app/database/db.py`):

```bash
# Windows PowerShell
$env:DATABASE_URL = "mysql+aiomysql://root:@127.0.0.1:3306/ai_highlighter"

# Command Prompt (CMD)
set DATABASE_URL=mysql+aiomysql://root:@127.0.0.1:3306/ai_highlighter
```

3. Install dependencies and run migrations:

```bash
python -m venv .venv 
source .venv/bin/activate          # Windows: .venv\Scripts\activate
.\.venv\Scripts\python -m pip install --upgrade pip #pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt #pip install -r requirements.txt
.\.venv\Scripts\python app\database\migrate.py #python3 app/database/migrate.py 
#pip install greenlet

winget install OpenJS.NodeJS.LTS #brew install node
winget install Gyan.FFmpeg #brew install ffmpeg

```

3. Run Lokal:
```bash
uvicorn app.main:app --reload
```
> Tip: Laragon’s default MySQL user is often `root` with an empty password. Adjust the URL if you have a password.

## Next steps
- Wire FastAPI app and routes
- Implement services for YouTube, transcripts, and highlighting
