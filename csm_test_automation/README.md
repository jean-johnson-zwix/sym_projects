# SYM CMS Telegram Testing Agent

Automated testing tool for the **Speak Your Mind (SYM)** Content Manager System. It simulates a real user interacting with the SYM Telegram bot (`@symassistanbot`), logs the full conversation, and uses Gemini AI to read bot responses, decide the correct replies, and score the output quality at each step.

---

## What It Does

- Connects to Telegram as a real user account via Telethon
- Sends messages to the SYM bot following a defined session plan (theme, post count, subthemes, constraints)
- Uses **Gemini AI** to read every bot response and decide the correct reply — no hardcoded keyword matching
- Handles the bot's memory (it may skip steps based on previous sessions)
- Waits for the bot to finish sending multiple messages before replying (idle timeout)
- Scores each bot response (1–10) across theme relevance, content quality, workflow correctness, and brand safety
- Logs everything to a single timestamped file ready to paste into Claude chat for report generation

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.9+ |
| Telegram client | [Telethon](https://github.com/LonamiWebs/Telethon) (MTProto user client) |
| AI decision engine | [Google Gemini](https://ai.google.dev/) via `google-genai` SDK (`gemini-2.5-flash`) |
| Environment config | `python-dotenv` |
| Session auth | Telethon `.session` file (one-time phone verification) |

---

## Setup

1. Create `.env` with your credentials:
   ```
   TELEGRAM_API_ID=your_api_id
   TELEGRAM_API_HASH=your_api_hash
   TELEGRAM_PHONE=+1234567890
   BOT_USERNAME=@symassistanbot
   GEMINI_API_KEY=your_gemini_api_key
   ```

2. Install dependencies:
   ```bash
   pip install telethon python-dotenv google-genai
   ```

3. Run a session:
   ```bash
   python tester.py --session 1
   ```

---

## Output

Each session produces a log file in `logs/` (e.g. `logs/session_1_brotherhood.txt`) containing every message sent and received, Gemini's step classification, and per-response scores. Feed this log to Claude chat to generate the final test report.

