# symm.city bot 🏙️

[@symmcitybot](https://t.me/symmcitybot) | [symm.city](https://symm.city)

A powerful, modular Telegram bot designed for the **symm.city** community. Built with Python, `python-telegram-bot` (v22+), FastAPI, PostgreSQL (`asyncpg`), and Google Gemini (`gemini-3.7-flash`).

---

## 🌟 Features

### 🛡️ Community Management & Moderation
- **Welcome Verification**: Greets new users with community rules, restricts their permissions until they interact with a verification button, and automatically kicks unverified users after 5 minutes to prevent spam.
- **Service Message Cleaner**: Automatically deletes noisy Telegram service messages (e.g. "User joined the group", "Message pinned") to keep the chat clean.
- **Admin Alerts**: Users can quickly ping all group administrators in emergencies by simply mentioning `@admin` in the chat.
- **Moderation Tools**: Standard commands for community management: `/mute`, `/unmute`, `/kick`, `/ban`.

### 🎮 CXP (Community Experience Points) System
- **Earn Points**: Users earn `50 CXP` for chatting (limited to 1 message per minute).
- **Reaction Economy**: Active members earn `+50 CXP` from positive reactions on their messages, or lose `-50 CXP` from negative reactions. 
- **Levels & Influence**: Users level up based on CXP. Higher-level users multiply the CXP effect of their reactions!
     * *Lvl 1-4:* Script Kiddie | *Lvl 5-9:* Hash Cracker
     * *Lvl 10-19:* True Operator | *Lvl 20-29:* Dirty Phreak
     * *Lvl 30-39:* Ledger Forger | *Lvl 40-49:* Clean Splicer
     * *Lvl 50-59:* Whale Hunter | *Lvl 60+:* Zero-Day Broker
- **Leaderboards**: View the top active members globally (`/leaderboard`) or filter by timeframe (`/leaderboard day`, `week`, `month`).
- **Contests**: Admins can use `/setcontest` to define a specific date range for special events, trackable via `/contest`.
- **Mini-Games**: Users can use `/steal` to attempt to steal 25-100 CXP from a random user (1-hour cooldown).

### 🤖 Grounded AI Assistant (`/ask`)
- **Grounded in Bluesky History**: Grounded in symmetricalboy's full Bluesky post archive using BM25 semantic retrieval, plus core knowledge of symm's projects (Symmetris, music, software, tech).
- **Powered by Gemini 3.7 Flash**: High-speed, accurate responses tailored to community inquiries.
- **Clean Chat**: Automatically cleans up the command and response after 60 seconds to avoid cluttering discussion topics.

### 🌐 Translation & Internationalization
- **Quick Translation**: Start or reply to a message with a language code command to instantly translate it (e.g. `/en`, `/es`, `/fr`, `/pt`, `/id`, `/ru`, `/tr`, `/fa`, `/uk`).
- **Interactive Menu**: Reply to any message with `/translate` to open an inline button menu with language options.
- **Smart Formatting**: Powered by `gemini-3.7-flash`, preserving Telegram HTML tags and original author attribution.

### ⏰ Time Zone Management
- **Check Times**: Easily check the exact time for a specific location (`/time London`), symm.city default (`/time`), or another user (`/time @username`).
- **Set Local Time**: Users can configure their personal time zone using `/settime <location>`.

### 📢 Channel Posting & Forwarding
- **Drafting Posts**: Designated channel admins can DM the bot (`/draft`) to compose, preview, and attach markdown URL buttons (`[Label](url) | [Label](url)`) before posting to the public channel.
- **Auto-Forwarding**: Prompts to forward channel posts directly to a designated discussion topic in the main group.

### 👑 Owner DM Navigation Panel
- **Setup Wizard**: When the bot owner first DMs the bot (`/start`), they are guided through an interactive setup process to link their main group, announcement channel, and admin topic.
- **Bot Configuration**: Inline menu to change group mappings, customize the welcome message, rules, and verify bot permissions across chats.
- **Channel Admin & Blacklist Management**: Add/remove authorized channel post authors and leaderboard blacklist entries.

---

## 🛠 Prerequisites

- **Python 3.14+** & [Poetry](https://python-poetry.org/docs/#installation)
- **PostgreSQL Database**
- **Telegram Bot Token** (obtain from [@BotFather](https://t.me/BotFather))
- **Google Gemini API Key** (for `gemini-3.7-flash` AI & translation features)

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/symmetricalboy/symm-city-bot.git
cd symm-city-bot
```

### 2. Backend Setup
```bash
# Navigate to the backend directory
cd backend

# Install dependencies using Poetry
poetry install
```

#### Environment Variables
Copy the `.env.example` file to create your local `.env` configuration file:
```bash
cp .env.example .env
```
Configure the required variables in `.env`:
```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
BOT_OWNER_ID=your_telegram_user_id_here
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
GEMINI_API_KEY=your_gemini_api_key_here
PORT=3000 # Default local port for FastAPI
```

*(Optional)* `PUBLIC_DOMAIN` or `RAILWAY_PUBLIC_DOMAIN` can be set if you are using webhooks instead of long polling.

### 3. Interactive Bot Setup
1. Start a Private Chat (DM) with your bot using the Telegram account specified in `BOT_OWNER_ID`.
2. Send `/start` to the bot.
3. The Setup Wizard will prompt for the `@usernames` of your Main Group, Channel, and Admin Group, as well as the link to your CXP topic.
4. After completing setup, the Owner Dashboard becomes available.

---

## 🏃 Running the Bot

### Local Development
```bash
poetry run python backend/bot.py
```

### Production Deployment (Railway)
1. Connect this repository to [Railway](https://railway.app/).
2. Provision a PostgreSQL add-on.
3. Set environment variables in Railway (`TELEGRAM_BOT_TOKEN`, `BOT_OWNER_ID`, `GEMINI_API_KEY`).
4. Railway will automatically build and start the bot using `railway.toml`.

---

*Developed for [symm.city](https://symm.city) by [@symmetricalboy](https://symmetricalboy.com)*
