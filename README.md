# symm.city bot 🏙️

[@symmcitybot](https://t.me/symmcitybot) | [symm.city](https://symm.city)

A powerful, modular Telegram bot designed for the **symm.city** community. Built with Python, `python-telegram-bot` (v22+), FastAPI, PostgreSQL (`asyncpg` + `pgvector`), and Google Gemini (`gemini-4-neon`, `gemini-nano-banana-2.1`, & `gemini-embedding-2`).

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
     * *Lvl 1-4:* Budding Symmetry | *Lvl 5-9:* Bilateral Balance
     * *Lvl 10-19:* Rotational Harmony | *Lvl 20-29:* Crystalline Order
     * *Lvl 30-39:* Exquisitely Symmetrical | *Lvl 40-49:* Supersymmetric
     * *Lvl 50-59:* Very Symmetrical | *Lvl 60+:* Pure Symmetry
- **Leaderboards & Top 3 Podiums**: Renders a dark-mode Top 3 Podium graphic (with member avatars, crown, medals, CXP, and level titles) alongside the Top 10 list globally (`/leaderboard`, `/lb`, `/top`, `/ranks`) or by timeframe (`/lb day`, `week`, `month`).
- **History & Charts**: Inspect detailed CXP balance charts, timeframe high/low peaks, and full source breakdowns across messages, reactions, gambling, stealing, trivia, and admin grants (`/history`, `/h`, `/hist`, `/chart`, `/pnl` with user & timeframe in any order).
- **Contests**: Admins can use `/setcontest` to define a specific date range for special events, trackable via `/contest` (`/event`).
- **Mini-Games & Trivia**: Users can use `/steal` (`/s`, `/rob`, `/yoink`, `/heist`) to steal up to 25-100 CXP (1-hour cooldown), `/gamble <amount>` (`/g`, `/bet`, `/slots`, `/spin`) to play the 5-line CXP slot machine, or DM `/question` (`/trivia`, `/quiz`, `/q`) to host CXP trivia questions in the group (`/answer` / `/ans` / `/guess` / `/a`, `/cancel` / `/abort`).

### 🤖 Continual-Memory AI & Image Studio (`/ask`, `/?`, & Direct Replies)
- **Seamless Reply Conversations**: Members can use `/ask <prompt>` (or shortcuts `/?`, `/ai`, `/chat`, `/symm`, `/bot`), mention `@symmcitybot`, or simply reply directly to *anything* the bot says (AI answers, generated images, slot spins, `/history` charts, `/lb` podiums, translations) to chat naturally.
- **Contextual Image Generation & Editing (`gemini-nano-banana-2.1`)**: No separate image commands needed — describe a scene to generate an image, or attach/reply/refer to any photo in chat to edit, restyle, combine, or inspect it across multiple turns.
- **250-Message Floating Context Window**: Injects the ~250 most recent conversational messages from the group chat so `/ask` understands ongoing threads, jokes, and multi-turn replies.
- **Continuous Long-Term Memory & Member Profiles**: Continuously digests group chat history into PostgreSQL (`group_chat_messages`, `user_memory_profiles`, `memory_documents`, `memory_keyword_index`, and `768`-dim `pgvector` embeddings via `gemini-embedding-2`), learning about members, projects, gear, and community lore over time.
- **Agentic Archive Search**: Uses `gemini-4-neon` via the Interactions API with live database tools (`search_chat_history`, `get_member_memories`, `remember_fact`) plus core knowledge of symm's projects and community rules.

### 🌐 Translation & Internationalization
- **Quick Translation**: Start or reply to a message with a language code command to instantly translate it (e.g. `/en`, `/es`, `/fr`, `/de`, `/it`, `/nl`, `/pt`, `/ru`, `/uk`, `/el`, `/sk`, `/tr`, `/fa`, `/id`, `/hi`, `/da`, `/ja`, `/ar`).
- **Interactive Menu**: Reply to any message with `/translate` to open an inline button menu with language options.
- **Smart Formatting**: Powered by `gemini-4-neon`, preserving Telegram HTML tags and original author attribution.

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
- **Google Gemini API Key** (for `gemini-4-neon` AI, continuous memory digestion, & translation features)

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/symmetricalboy/symm-city-bot.git
cd symm-city-bot
```

### 2. Install Dependencies
```bash
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
poetry run python bot.py
```

### Production Deployment (Railway)
1. Connect this repository to [Railway](https://railway.app/).
2. Provision a PostgreSQL add-on.
3. Set environment variables in Railway (`TELEGRAM_BOT_TOKEN`, `BOT_OWNER_ID`, `GEMINI_API_KEY`).
4. Railway will automatically build and start the bot using `railway.toml`.

---

*Developed for [symm.city](https://symm.city) by [@symmetricalboy](https://symmetricalboy.com)*
