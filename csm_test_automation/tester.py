"""
SYM CMS Telegram Testing Agent
Uses Gemini to read bot responses, decide replies, and score output quality.

Usage:
    python tester.py --session 1
    python tester.py --session 4 --log logs/custom.txt --report reports/custom.md
"""

import asyncio
import argparse
import json
import os
import time
from datetime import datetime

from google import genai
from dotenv import load_dotenv
from telethon import TelegramClient

load_dotenv()

# --- Credentials ---
API_ID         = int(os.getenv("TELEGRAM_API_ID"))
API_HASH       = os.getenv("TELEGRAM_API_HASH")
PHONE          = os.getenv("TELEGRAM_PHONE")
BOT_USERNAME   = os.getenv("BOT_USERNAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

BOT_POLL_INTERVAL = 20   # seconds between polls
BOT_TIMEOUT       = 300  # max seconds to wait for the first bot message
BOT_IDLE_TIMEOUT  = 45   # seconds of silence before considering bot done sending


# ---------------------------------------------------------------------------
# Session plans
# ---------------------------------------------------------------------------

SESSION_PLANS = {
    1: {
        "theme": "Brotherhood Among Men",
        "posts": 1,
        "type": "happy_path",
        "subthemes": "shared growth, sharing",
        "avoid": None,
        "log": "logs/session_1_brotherhood.txt",
    },
    2: {
        "theme": "Healing",
        "posts": 1,
        "type": "vague",
        "subthemes": None,
        "avoid": None,
        "log": "logs/session_2_healing.txt",
    },
    3: {
        "theme": "Overcoming Anxiety",
        "posts": 3,
        "type": "multi_post",
        "subthemes": "daily life, small wins, progress",
        "avoid": None,
        "log": "logs/session_3_anxiety.txt",
    },
    4: {
        "theme": "Finding Your Voice",
        "posts": 1,
        "type": "rejection_flow",
        "subthemes": "self-expression, speaking up",
        "avoid": None,
        "log": "logs/session_4_finding_voice.txt",
    },
    5: {
        "theme": "Men and Mental Health at Work",
        "posts": 2,
        "type": "constrained",
        "subthemes": "burnout, asking for help, workplace pressure",
        "avoid": "framing vulnerability or asking for help as weakness",
        "log": "logs/session_5_men_mental_health.txt",
    },
}


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class SessionLogger:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path

    def _write(self, tag: str, text: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{tag}] {text}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def sent(self, text: str):       self._write("SENT", text)
    def received(self, text: str):   self._write("RECV", text)
    def info(self, text: str):       self._write("INFO", text)
    def score(self, text: str):      self._write("SCORE", text)
    def anomaly(self, text: str):    self._write("ANOMALY", text)


# ---------------------------------------------------------------------------
# Gemini agent
# ---------------------------------------------------------------------------

APPROVAL_INSTRUCTIONS = {
    "happy_path": (
        "Approve everything. No rejections. Single post."
    ),
    "vague": (
        "The theme is intentionally broad with no subthemes. Approve everything. "
        "Note in score_notes if the bot asks for clarification or handles ambiguity well."
    ),
    "multi_post": (
        "Requesting 3 posts. Approve all ideas. Approve short captions. Approve all image prompts."
    ),
    "rejection_flow": (
        "IDEA: Reject the very first idea the bot presents. "
        "Say: 'Please focus more on speaking up in personal relationships'. "
        "Then approve the next idea it offers.\n"
        "CAPTION: Reject the very first caption. "
        "Say: 'Please make it shorter and more direct'. Then approve the next one.\n"
        "IMAGE PROMPT: Approve normally."
    ),
    "constrained": (
        "Subthemes to include: burnout, asking for help, workplace pressure. "
        "Subthemes to AVOID: framing vulnerability or asking for help as weakness. "
        "Approve both ideas. Approve short captions. Approve all image prompts. "
        "Flag in score_notes if the bot violates the avoid constraint."
    ),
}

GEMINI_SYSTEM_PROMPT = """You are an AI testing agent for the Speak Your Mind (SYM) Content Manager System.
SYM is a mental health and emotional wellness community for men.
You are testing a Telegram bot that helps create social media posts.

IMPORTANT: The bot has memory of previous sessions. It may skip steps and jump straight
to presenting ideas right after "hi", without asking for theme or post count first.
You must handle whatever stage the bot is currently at — do not assume a fixed order.

The bot's typical workflow (may be partial or reordered due to memory):
  greeting → (ask theme) → (ask post count) → present ideas → approve/adjust idea
  → present caption options → approve/adjust caption
  → present image prompt → approve/adjust prompt → deliver final output

YOUR RULES:
- If the bot asks for the theme: reply with the theme from the SESSION PLAN (include subthemes/avoid if set).
- If the bot asks for post count: reply with the number from the SESSION PLAN.
- If the bot presents ideas and asks for a choice: follow APPROVAL INSTRUCTIONS.
- If the bot presents captions: follow APPROVAL INSTRUCTIONS.
- If the bot presents an image prompt: follow APPROVAL INSTRUCTIONS.
- If the bot's response is a general greeting or menu with no specific ask: reply with the theme to move things forward.
- If the bot presents an image prompt or asks to approve/generate an image: set step=prompt_approval and reply with "Approve" (or per APPROVAL INSTRUCTIONS). Do NOT set is_complete=true yet.
- If the bot explicitly confirms the full process is done (e.g. "your post is ready", "process complete", "final output", "all set") AND has already gone through idea + caption + image prompt: set is_complete=true.
- NEVER set is_complete=true just because a caption was delivered. The image prompt step must come after captions.

SCORING (1–10) — score every substantive bot response on:
  - Theme relevance: Does the content match the requested theme?
  - Content quality: Well-written and emotionally appropriate for a men's wellness audience?
  - Workflow correctness: Is the bot doing the right thing at this stage?
  - Brand safety: Free of harmful, clinical, or off-brand language?
  Skip scoring (score=0) only for pure greeting/navigation messages with no content.

Return ONLY valid JSON — no markdown, no extra text:
{
  "step": "<greeting|theme_prompt|post_count_prompt|idea_approval|caption_approval|prompt_approval|complete|unknown>",
  "reply": "<your reply to send to the bot>",
  "score": <integer 0-10>,
  "score_notes": "<one or two sentences explaining the score, or empty string if score=0>",
  "is_complete": <true|false>
}"""


class GeminiAgent:
    def __init__(self, plan: dict):
        self.plan = plan
        self.history: list[dict] = []  # {"role": "bot"|"tester", "text": str}

    def record_sent(self, text: str):
        self.history.append({"role": "tester", "text": text})

    def decide(self, bot_message: str) -> dict:
        """Feed the bot message to Gemini and return its decision."""
        self.history.append({"role": "bot", "text": bot_message})

        history_str = "\n".join(
            f"{'BOT' if h['role'] == 'bot' else 'TESTER'}: {h['text']}"
            for h in self.history
        )

        plan = self.plan
        approval_instructions = APPROVAL_INSTRUCTIONS.get(plan["type"], "Approve everything.")

        prompt = f"""{GEMINI_SYSTEM_PROMPT}

SESSION PLAN:
- Theme: {plan['theme']}
- Posts requested: {plan['posts']}
- Session type: {plan['type']}
- Subthemes to include: {plan.get('subthemes') or 'None'}
- Subthemes to avoid: {plan.get('avoid') or 'None'}

APPROVAL INSTRUCTIONS:
{approval_instructions}

FULL CONVERSATION SO FAR:
{history_str}

Based on the above, respond with the JSON decision now."""

        max_retries = 5
        wait = 10  # seconds, doubles each retry
        result = None
        for attempt in range(1, max_retries + 1):
            try:
                response = gemini.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                raw = response.text.strip()
                if raw.startswith("```"):
                    parts = raw.split("```")
                    raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
                result = json.loads(raw)
                break
            except Exception as e:
                err = str(e)
                if attempt < max_retries and ("503" in err or "UNAVAILABLE" in err or "429" in err):
                    print(f"[GEMINI] Attempt {attempt} failed ({err[:80]}). Retrying in {wait}s...")
                    time.sleep(wait)
                    wait *= 2
                else:
                    result = {
                        "step": "unknown",
                        "reply": "Could you clarify what you need from me at this step?",
                        "score": 0,
                        "score_notes": f"Gemini error: {e}",
                        "is_complete": False,
                    }
                    break

        if result.get("reply"):
            self.history.append({"role": "tester", "text": result["reply"]})

        return result


# ---------------------------------------------------------------------------
# Telegram session runner
# ---------------------------------------------------------------------------

class SessionRunner:
    def __init__(self, client: TelegramClient, plan: dict,
                 logger: SessionLogger, agent: GeminiAgent):
        self.client  = client
        self.plan    = plan
        self.logger  = logger
        self.agent   = agent
        self.issues: list[dict] = []
        self.steps_completed: list[str] = []
        self._last_seen_id = 0

    async def _latest_bot_msg_id(self) -> int:
        msgs = await self.client.get_messages(BOT_USERNAME, limit=10)
        for m in msgs:
            if not m.out:
                return m.id
        return 0

    async def send(self, text: str):
        """Snapshot last bot message ID, then send."""
        self._last_seen_id = await self._latest_bot_msg_id()
        self.logger.sent(text)
        self.agent.record_sent(text)
        await self.client.send_message(BOT_USERNAME, text)

    async def wait_for_response(self) -> str:
        """
        Wait for the bot to finish sending all its messages.
        - Polls until the first new message arrives (up to BOT_TIMEOUT).
        - Then keeps collecting until BOT_IDLE_TIMEOUT seconds pass with no new message.
        - Returns all messages joined with a newline separator.
        """
        # Phase 1: wait for the first new message
        elapsed = 0
        while elapsed < BOT_TIMEOUT:
            await asyncio.sleep(BOT_POLL_INTERVAL)
            elapsed += BOT_POLL_INTERVAL
            msgs = await self.client.get_messages(BOT_USERNAME, limit=20)
            new = [m for m in reversed(msgs) if not m.out and m.id > self._last_seen_id]
            if new:
                for m in new:
                    self._last_seen_id = max(self._last_seen_id, m.id)
                collected = [m.text or "" for m in new if m.text]
                break
        else:
            return ""  # timed out waiting for first message

        # Phase 2: keep collecting until idle for BOT_IDLE_TIMEOUT seconds
        idle = 0
        while idle < BOT_IDLE_TIMEOUT:
            await asyncio.sleep(BOT_POLL_INTERVAL)
            idle += BOT_POLL_INTERVAL
            msgs = await self.client.get_messages(BOT_USERNAME, limit=20)
            new = [m for m in reversed(msgs) if not m.out and m.id > self._last_seen_id]
            if new:
                for m in new:
                    self._last_seen_id = max(self._last_seen_id, m.id)
                collected += [m.text or "" for m in new if m.text]
                idle = 0  # reset idle timer on each new message

        return "\n\n---\n\n".join(collected)

    async def _send_and_wait(self, text: str) -> str:
        """Send a message, then block until the bot replies."""
        await self.send(text)
        response = await self.wait_for_response()
        if response:
            self.logger.received(response)
        return response

    async def run(self):
        logger = self.logger
        plan   = self.plan
        logger.info(f"Session start — Theme: {plan['theme']} | Type: {plan['type']} | Posts: {plan['posts']}")

        # Kick off with "hi" — Gemini handles everything from the first response onward
        await self.send("hi")

        # --- Main Gemini-driven loop ---
        max_turns = 40
        turn = 0

        while turn < max_turns:
            turn += 1

            bot_response = await self.wait_for_response()
            if not bot_response:
                self.issues.append({
                    "step": "Unknown",
                    "what": "Bot did not respond within timeout.",
                    "severity": "High",
                    "owner": "Hector",
                })
                logger.anomaly("Bot timed out. Ending session.")
                break

            logger.received(bot_response)

            # Ask Gemini
            decision    = self.agent.decide(bot_response)
            step        = decision.get("step", "unknown")
            reply       = decision.get("reply", "")
            score       = decision.get("score", 0)
            score_notes = decision.get("score_notes", "")
            is_complete = decision.get("is_complete", False)

            logger.score(f"step={step} | {score}/10 — {score_notes}")

            # Track workflow steps
            step_label_map = {
                "idea_approval":    "Idea",
                "caption_approval": "Caption",
                "prompt_approval":  "Image Prompt",
                "complete":         "Final Output",
            }
            if step in step_label_map:
                label = step_label_map[step]
                if label not in self.steps_completed:
                    self.steps_completed.append(label)

            if is_complete or step == "complete":
                # Only close if image prompt step was already seen; otherwise keep going
                if "Image Prompt" not in self.steps_completed:
                    logger.info("Gemini signalled complete but Image Prompt not yet seen — continuing.")
                    if reply:
                        await self.send(reply)
                    continue
                if "Final Output" not in self.steps_completed:
                    self.steps_completed.append("Final Output")
                logger.info("Session complete.")
                break

            if reply:
                await self.send(reply)

        return self.steps_completed, self.issues




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SYM CMS Telegram Tester")
    p.add_argument("--session", type=int, choices=[1, 2, 3, 4, 5], required=True)
    p.add_argument("--log", type=str, help="Override log file path")
    return p.parse_args()


async def main():
    args = parse_args()
    plan = dict(SESSION_PLANS[args.session])
    if args.log:
        plan["log"] = args.log

    logger = SessionLogger(plan["log"])
    agent  = GeminiAgent(plan)

    client = TelegramClient("sym_tester", API_ID, API_HASH)
    await client.start(phone=PHONE)
    logger.info("Telegram client authenticated.")

    runner = SessionRunner(client, plan, logger, agent)
    steps_completed, issues = await runner.run()

    await client.disconnect()
    logger.info(f"Session done. Steps: {' → '.join(steps_completed) if steps_completed else 'None'}")
    logger.info(f"Log saved to: {plan['log']}")


if __name__ == "__main__":
    asyncio.run(main())
