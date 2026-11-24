# Telegram Event Deadline Reminder Bot

A production-ready, async Python Telegram Bot worker (`python-telegram-bot` v21+, `BOT_TOKEN` only) that monitors a Telegram channel, detects event deadlines with the Google Gemini API (`google-genai` structured JSON output), persists all state in **Supabase PostgreSQL** (`asyncpg`), and runs cleanly as a background worker on **Railway**.

---

## Features

- **Bot Token Only:** Uses `python-telegram-bot` v21+ exclusively with `BOT_TOKEN` — no Telethon, Pyrogram, API ID, API Hash, or user session files.
- **Strict Private Admin-Only Access:** A global pre-handler gatekeeper (`group=-1`) blocks every non-admin user and non-private interaction immediately (`ApplicationHandlerStop`). Unauthorized users trying `/start`, any command, messages, or inline buttons are rejected with zero access to menus, data, or actions.
- **Automatic Channel Monitoring & Gemini Extraction:**
  - Listens to `channel_post` and `edited_channel_post` updates on your configured `CHANNEL_ID`.
  - Caches post text in Supabase (`channel_posts_cache`) for manual link lookups.
  - Uses `google-genai` (`gemini-4-hydrogen` default with automatic `503 UNAVAILABLE` / transient retry + backoff and automatic fallback across `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, and `gemini-4-argon`, configurable via `GEMINI_MODEL` and `GEMINI_FALLBACK_MODELS`) with Pydantic `response_schema` structured JSON output and `Asia/Dhaka` date resolution.
  - Ignores already-expired historical posts and constructs canonical public (`https://t.me/<username>/<msg_id>`) and private (`https://t.me/c/<internal_id>/<msg_id>`) links.
- **Multi-Step Admin Confirmation Flow:**
  - Sends detected deadlines privately to `ADMIN_USER_ID` with **Confirm** / **Decline** inline buttons.
  - **Step 1 (Reminder Window):** `1 Day`, `2 Days`, `3 Days`, `5 Days`, `7 Days`, `10 Days` (with a safe **Back** button returning to the detection prompt).
  - **Step 2 (Repeat Mode):**
    - **Normal (Daily):** 1 reminder per day while `1 <= days_left <= window_days`.
    - **Alternate Day:** 1 reminder every 2 days starting from the first day of the window (`(window_days - days_left) % 2 == 0`, e.g. `10 Days` → `10, 8, 6, 4, 2`).
- **Admin Control Panel (`/start` or `/admin`):**
  - **Active Reminders:** Paginated list showing Title, End Date, Window, and Mode, with Overview and Delete actions.
  - **Instant Reminder:** Select any active reminder by Title to open its **Full Event Overview**, then **Send Instant Reminder**, **Edit Settings**, **Delete Reminder**, or go **Back to List**.
  - **Set Reminder Post (Manual Link Entry):** Paste any public or private post link → uses Supabase post cache or accepts a forwarded post / pasted text → Gemini extraction with **Missing End Date Fallback** (`YYYY-MM-DD`, `DD-MM-YYYY`, or `31 December 2026`, rejecting past dates) and Title fallback.
- **Atomic Scheduler (`APScheduler` + Supabase PostgreSQL):**
  - Runs daily at `10:00 AM Asia/Dhaka` (configurable), plus an hourly deduplicated safety check and startup catch-up.
  - Uses atomic PostgreSQL row claims (`FOR UPDATE` + `last_reminded_on <> today`) so restarts or concurrent runs never send duplicate reminders and failed deliveries automatically revert and retry.
  - Marks reminders `expired` automatically once `days_left < 0`.

---

## Project Structure

All required files live neatly in the project root for zero-friction GitHub upload and Railway deployment:

```text
telegram-deadline-reminder-bot/
├── main.py             # Railway worker entrypoint, config validation, graceful shutdown
├── database.py         # Async Supabase PostgreSQL (asyncpg) pool, schema, atomic claims & CRUD
├── gemini_parser.py    # Google GenAI SDK structured output (response_schema) with retries
├── scheduler.py        # APScheduler (AsyncIOScheduler) jobs, eligibility math, channel poster
├── handlers.py         # Strict admin gatekeeper, channel monitor, /admin panel, manual flow
├── keyboards.py        # InlineKeyboardMarkup builders for all menus and multi-step flows
├── requirements.txt    # Python dependencies (asyncpg, python-telegram-bot, google-genai, etc.)
├── railway.toml        # Railway worker start command and restart policy
├── Procfile            # Worker process definition (`worker: python -u main.py`)
├── runtime.txt         # Non-dot Python 3.12 pin for drag-and-drop Railway deploys
├── .python-version     # Standard Python 3.12 version pin
├── .env.example        # Environment variable template (safe to commit)
├── .gitignore          # Excludes .env, __pycache__, venvs, local DBs, and temp files
└── README.md           # Setup, GitHub drag-and-drop upload, Supabase, and Railway guide
```

---

## Setup Guide

### 1. Supabase Database Setup
1. Create a free project at [Supabase](https://supabase.com/).
2. Open **Project Settings → Database → Connection string → URI**.
3. Copy the **Transaction Pooler** URI (recommended for Railway because it supports IPv4 out of the box; port `6543`) or the Direct URI (port `5432`).
   - Example:
     `postgresql://postgres.<project-ref>:<password>@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres`
   - If your database password contains special characters (`@`, `#`, `:`, `/`, `%`), URL-encode them.
4. Tables (`reminders`, `channel_posts_cache`) and indexes are created automatically when the worker boots.

### 2. Telegram Bot & Channel Setup
1. Create a bot with [@BotFather](https://t.me/BotFather) and copy `BOT_TOKEN`.
2. Add the bot as an **Administrator** to your target Telegram channel with **Post Messages** permission.
3. Get your numeric Telegram User ID (`ADMIN_USER_ID`) via [@userinfobot](https://t.me/userinfobot).
4. Start a private chat with your bot and send `/start` so Telegram allows private notifications.

---

## Uploading to GitHub (Web Drag-and-Drop — Single Batch)

A pre-cleaned **`github_upload/`** staging folder is provided inside the project folder so you can upload everything in **one drag-and-drop action** without accidentally including `.env`, caches, or temporary files — and it still works cleanly on Railway even if your file manager hides dotfiles.

1. Create a new **Private** (or Public) repository on GitHub.
2. On the GitHub repository page, click **Add file → Upload files** (or **uploading an existing file** on an empty repo).
3. Open the **`github_upload`** folder on your computer:
   - *(Recommended)* Enable hidden files so `.gitignore`, `.env.example`, and `.python-version` are also included:
     - **Windows File Explorer:** `View → Show → Hidden items`
     - **macOS Finder:** `Cmd + Shift + .`
   - Even if hidden files remain off, `runtime.txt`, `railway.toml`, `Procfile`, `requirements.txt`, `env.example.txt`, and all Python modules are regular non-dot files and deploy cleanly on Railway with zero extra setup.
4. Open **inside** `github_upload/`, press **`Ctrl + A`** (Windows) or **`Cmd + A`** (macOS) to select all files inside it, and drag-and-drop them directly into GitHub's upload box.
   > **Important:** Drag the **files inside** `github_upload/`, not the `github_upload` folder itself, so `main.py` and `requirements.txt` land in the repository root.
5. Click **Commit changes**.

*(Never upload `.env`, `__pycache__/`, `.venv/`, or local `.db`/`.log` files.)*

---

## Deploying as a Worker on Railway

1. In [Railway](https://railway.com/), click **New Project → Deploy from GitHub repo** and select your repository.
2. Railway automatically reads `railway.toml`, `Procfile`, `.python-version`, and `requirements.txt` to build and launch the bot as a background worker (`python -u main.py`) with **no public HTTP domain required**.
3. Open the service's **Variables** tab and add:
   - `BOT_TOKEN`
   - `GEMINI_API_KEY`
   - `ADMIN_USER_ID`
   - `CHANNEL_ID`
   - `DATABASE_URL` *(your Supabase PostgreSQL URI)*
   - `TIMEZONE` = `Asia/Dhaka`
   - *(Optional)* `GEMINI_MODEL` = `gemini-4-hydrogen`, `GEMINI_FALLBACK_MODELS` = `gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-4-argon`, `DAILY_REMINDER_HOUR` = `10`, `DAILY_REMINDER_MINUTE` = `0`
4. Keep **Replicas = 1** (Telegram long-polling allows only one active `getUpdates` poller per bot token).
5. Check **Deploy Logs** — you should see:
   - `Connected to Supabase PostgreSQL database pool`
   - `ReminderScheduler started (daily=10:00 Asia/Dhaka, plus hourly deduplicated check)`
   - `Starting Telegram Deadline Reminder Worker ...`

---

## Local Development

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Fill in BOT_TOKEN, GEMINI_API_KEY, ADMIN_USER_ID, CHANNEL_ID, and DATABASE_URL
python -u main.py
```

---

## Channel Reminder Format

Both automated and instant reminders are posted to your channel in a clean, bold, emoji-rich HTML layout (safely HTML-escaped and capped within Telegram's 4096-character limit):

```text
⏰ EVENT DEADLINE REMINDER
━━━━━━━━━━━━━━━━━━━━

📌 Winter AI Hackathon 2026

🔥 1 Day Left — Ends Tomorrow!
🗓 Deadline: 2026-12-10 (Asia/Dhaka)
🔁 Schedule: Last 5 Days • Normal (Daily)

🔗 Event Post: Open Original Post ↗

━━━━━━━━━━━━━━━━━━━━
✨ Don't miss out — complete your action before the deadline closes!
```
