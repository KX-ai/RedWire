"""
RedWire AI — Manchester United Football Chatbot
Pipeline: User Query → Agent (Intent Extraction) → API Fetch → RAG → LLM Response
"""

import streamlit as st
import requests
import json
import os
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG — st.secrets (Streamlit Cloud) with .env fallback
# ─────────────────────────────────────────────
def _secret(key: str, default: str = None) -> str:
    """Read from st.secrets first, then os.getenv."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

RAPIDAPI_KEY = _secret("RAPIDAPI_KEY")
GROQ_API_KEY = _secret("GROQ_API_KEY")
GROQ_MODEL   = _secret("GROQ_MODEL", "openai/gpt-oss-120b")

groq_client = Groq(api_key=GROQ_API_KEY)

FOOTBALL_HOST = "free-api-live-football-data.p.rapidapi.com"
FOOTBALL_BASE = f"https://{FOOTBALL_HOST}"

RAPIDAPI_HEADERS = {
    "x-rapidapi-key":  RAPIDAPI_KEY,
    "Content-Type":    "application/json",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _get(url: str, host: str, params: dict = None) -> dict | list | None:
    """Generic GET with error handling."""
    try:
        headers = {**RAPIDAPI_HEADERS, "x-rapidapi-host": host}
        r = requests.get(url, headers=headers, params=params, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


# ─────────────────────────────────────────────
# X / TWITTER  (primary → fallback)
# ─────────────────────────────────────────────
def search_x(query: str, n: int = 8) -> list[dict]:
    # Primary: twitter-api45
    data = _get(
        "https://twitter-api45.p.rapidapi.com/search.php",
        "twitter-api45.p.rapidapi.com",
        {"query": query, "search_type": "Top"},
    )
    if data and "_error" not in data:
        tweets = [
            {
                "username": t.get("screen_name"),
                "text":     t.get("text"),
                "likes":    t.get("favorites", 0),
                "retweets": t.get("retweets", 0),
                "replies":  t.get("replies", 0),
                "created_at": t.get("created_at"),
                "source":   "X",
            }
            for t in data.get("timeline", [])
            if t.get("type") == "tweet"
            and not str(t.get("tweet_id", "")).startswith("promoted-")
            and t.get("lang", "en") == "en"
        ]
        if tweets:
            return tweets[:n]

    # Fallback: twitter-api47
    data = _get(
        "https://twitter-api47.p.rapidapi.com/v3/search",
        "twitter-api47.p.rapidapi.com",
        {"query": query, "type": "Top"},
    )
    if data and "_error" not in data:
        return [
            {
                "username": t.get("author", {}).get("username"),
                "text":     t.get("text"),
                "likes":    t.get("likeCount", 0),
                "retweets": t.get("retweetCount", 0),
                "replies":  t.get("replyCount", 0),
                "created_at": t.get("createdAt"),
                "source":   "X",
            }
            for t in data.get("data", [])
            if not t.get("isPromoted", False)
            and t.get("type") in ("tweet", "quote")
            and t.get("lang", "en") == "en"
        ][:n]
    return []


# ─────────────────────────────────────────────
# THREADS  (primary → fallback)
# ─────────────────────────────────────────────
def _parse_threads(items: list, source_tag: str) -> list[dict]:
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("caption") or item.get("content") or ""
        if not text:
            continue
        out.append({
            "username": (
                item.get("username")
                or item.get("user", {}).get("username")
                or "unknown"
            ),
            "text":     text,
            "likes":    item.get("like_count") or item.get("likes") or 0,
            "replies":  item.get("reply_count") or item.get("replies") or 0,
            "created_at": item.get("created_at") or item.get("timestamp") or "",
            "source":   "Threads",
        })
    return out


def search_threads(query: str, n: int = 6) -> list[dict]:
    # Primary: threads-api4
    data = _get(
        "https://threads-api4.p.rapidapi.com/api/search/top",
        "threads-api4.p.rapidapi.com",
        {"query": query},
    )
    if data and "_error" not in data:
        items = (
            data.get("data")
            or data.get("posts")
            or data.get("threads")
            or (data if isinstance(data, list) else [])
        )
        posts = _parse_threads(items, "Threads")
        if posts:
            return posts[:n]

    # Fallback: threads-scraper
    data = _get(
        "https://threads-scraper.p.rapidapi.com/api/v1/posts/search",
        "threads-scraper.p.rapidapi.com",
        {"query": query},
    )
    if data and "_error" not in data:
        items = (
            data.get("data")
            or data.get("posts")
            or (data if isinstance(data, list) else [])
        )
        return _parse_threads(items, "Threads")[:n]
    return []


# ─────────────────────────────────────────────
# FOOTBALL API
# ─────────────────────────────────────────────
def _football(endpoint: str, params: dict = None) -> dict:
    return _get(f"{FOOTBALL_BASE}/{endpoint}", FOOTBALL_HOST, params)


@st.cache_data(ttl=3600, show_spinner=False)
def get_man_utd_id() -> str | None:
    """Resolve Man Utd team ID once per hour."""
    data = _football("football-teams-search", {"search": "Manchester United"})
    if not data or "_error" in data:
        return None
    teams = (
        data.get("response")
        or data.get("teams")
        or data.get("data")
        or (data if isinstance(data, list) else [])
    )
    for t in teams:
        name = (t.get("name") or t.get("teamName") or "").lower()
        if "manchester united" in name or "man utd" in name:
            tid = t.get("id") or t.get("teamId") or t.get("team_id")
            return str(tid) if tid else None
    return None


def get_live_scores() -> dict:
    return _football("football-live-scores")


def get_standings(competition: str = "PL") -> dict:
    return _football("football-current-season-standings", {"competitionId": competition})


def get_fixtures(team_id: str) -> dict:
    return _football("football-team-next-matches", {"teamId": team_id})


def get_last_matches(team_id: str) -> dict:
    return _football("football-team-last-matches", {"teamId": team_id})


def get_squad(team_id: str) -> dict:
    return _football("football-team-players", {"teamId": team_id})


# ─────────────────────────────────────────────
# AGENT — INTENT EXTRACTION
# ─────────────────────────────────────────────
INTENT_SYSTEM = """
You are a query router for a Manchester United football chatbot.

Given a user query, return ONLY a valid JSON object — no markdown, no explanation — with:
{
  "intent": one of ["live_scores","fixtures","results","standings","squad","transfers","injuries","social","general"],
  "needs_football_api": true/false,
  "needs_social_api": true/false,
  "social_keywords": ["keyword1", "keyword2"],
  "summary": "one-line description of what the user wants"
}

Rules:
- "live_scores"  → user asks about ongoing/today's matches
- "fixtures"     → upcoming matches, next game, schedule
- "results"      → past match scores, recent results
- "standings"    → league table, position, points
- "squad"        → players, lineup, squad list
- "transfers"    → transfer news, signings, departures
- "injuries"     → injury news, who is injured, fitness
- "social"       → what fans think, reactions, opinions, social media
- "general"      → anything else about Man Utd
- Always set needs_social_api=true for "transfers", "injuries", "social", "general"
- Always set needs_football_api=true for "live_scores","fixtures","results","standings","squad"
- social_keywords: 2-4 search terms for X/Threads related to the query
"""

def extract_intent(user_query: str) -> dict:
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user",   "content": user_query},
            ],
            temperature=0.1,
            max_tokens=250,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception:
        return {
            "intent": "general",
            "needs_football_api": True,
            "needs_social_api": True,
            "social_keywords": ["Manchester United"],
            "summary": user_query,
        }


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
def fetch_data(intent_data: dict, team_id: str | None) -> dict:
    intent   = intent_data.get("intent", "general")
    keywords = intent_data.get("social_keywords", ["Manchester United"])
    social_q = " ".join(keywords)
    result   = {}

    # ── Football API ──────────────────────────
    if intent_data.get("needs_football_api"):
        if intent == "live_scores":
            result["live_scores"] = get_live_scores()
        elif intent == "standings":
            result["standings"] = get_standings("PL")
        elif intent == "fixtures" and team_id:
            result["fixtures"] = get_fixtures(team_id)
        elif intent == "results" and team_id:
            result["results"] = get_last_matches(team_id)
        elif intent == "squad" and team_id:
            result["squad"] = get_squad(team_id)
        else:
            # General: pull standings + fixtures for context
            result["standings"] = get_standings("PL")
            if team_id:
                result["fixtures"] = get_fixtures(team_id)

    # ── Social APIs ───────────────────────────
    if intent_data.get("needs_social_api"):
        result["x_posts"]      = search_x(social_q)
        result["threads_posts"] = search_threads(social_q)

    return result


# ─────────────────────────────────────────────
# RAG — CONTEXT BUILDER
# ─────────────────────────────────────────────
def build_context(data: dict) -> str:
    parts = []

    if "live_scores" in data and "_error" not in (data["live_scores"] or {}):
        parts.append("=== LIVE SCORES ===\n" + json.dumps(data["live_scores"], indent=2)[:2500])

    if "standings" in data and "_error" not in (data["standings"] or {}):
        parts.append("=== PREMIER LEAGUE STANDINGS ===\n" + json.dumps(data["standings"], indent=2)[:2500])

    if "fixtures" in data and "_error" not in (data["fixtures"] or {}):
        parts.append("=== UPCOMING FIXTURES ===\n" + json.dumps(data["fixtures"], indent=2)[:2000])

    if "results" in data and "_error" not in (data["results"] or {}):
        parts.append("=== RECENT RESULTS ===\n" + json.dumps(data["results"], indent=2)[:2000])

    if "squad" in data and "_error" not in (data["squad"] or {}):
        parts.append("=== SQUAD / PLAYERS ===\n" + json.dumps(data["squad"], indent=2)[:2000])

    if data.get("x_posts"):
        lines = [
            f"@{p['username']}: {p['text']}  [❤ {p['likes']} | 🔁 {p.get('retweets',0)}]"
            for p in data["x_posts"]
        ]
        parts.append("=== FAN REACTIONS (X / Twitter) ===\n" + "\n".join(lines))

    if data.get("threads_posts"):
        lines = [
            f"@{p['username']}: {p['text']}  [❤ {p['likes']}]"
            for p in data["threads_posts"]
        ]
        parts.append("=== FAN REACTIONS (Threads) ===\n" + "\n".join(lines))

    return "\n\n".join(parts) if parts else "No live data retrieved for this query."


# ─────────────────────────────────────────────
# RESPONSE GENERATION (streaming)
# ─────────────────────────────────────────────
ANSWER_SYSTEM = """
You are RedWire AI — the ultimate Manchester United intelligence chatbot.
You are passionate, knowledgeable, and speak like a true Red Devil supporter.

Use the real-time data context below to answer the user's question accurately and engagingly.
- If context has relevant data, use it as the primary source of truth.
- If context is missing data, be honest but still give helpful background knowledge.
- Format answers clearly. Use emojis where natural. Keep it sharp and enthusiastic.
- Refer to the club as "United", "the Reds", or "the Red Devils".
- Never make up scores, dates, or player names.

Today: {today}

=== REAL-TIME DATA CONTEXT ===
{context}
"""

def stream_response(user_query: str, context: str):
    system = ANSWER_SYSTEM.format(
        today=datetime.now().strftime("%A, %d %B %Y %H:%M"),
        context=context,
    )
    return groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_query},
        ],
        temperature=0.7,
        max_tokens=900,
        stream=True,
    )


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(ellipse at top, #1a0000 0%, #0a0a0a 60%);
        min-height: 100vh;
    }

    /* Header */
    .rw-header {
        text-align: center;
        padding: 2rem 0 0.5rem;
    }
    .rw-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #DA291C 30%, #FBE122 70%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin: 0;
    }
    .rw-subtitle {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.3rem;
        letter-spacing: 0.5px;
    }

    /* Quick-action buttons */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #1c0000, #2a0000);
        border: 1px solid #DA291C44;
        color: #ccc;
        border-radius: 8px;
        font-size: 0.78rem;
        transition: all 0.2s;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #DA291C;
        color: #fff;
        background: linear-gradient(135deg, #2a0000, #3a0000);
        transform: translateX(2px);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.2rem;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: #111 !important;
        border: 1px solid #DA291C55 !important;
        border-radius: 12px !important;
        color: #fff !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #DA291C !important;
        box-shadow: 0 0 0 2px #DA291C22 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d0d0d;
        border-right: 1px solid #1f1f1f;
    }

    /* Status widget */
    [data-testid="stStatus"] {
        background: #111 !important;
        border: 1px solid #DA291C33 !important;
        border-radius: 10px !important;
    }

    /* Divider */
    hr { border-color: #1f1f1f !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #DA291C44; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #DA291C; }
    </style>
    """, unsafe_allow_html=True)


QUICK_QUESTIONS = [
    "⚽ What's United's next match?",
    "📊 Where are United in the table?",
    "🏆 What were United's last results?",
    "🔄 Latest transfer rumours?",
    "🤕 Any injury news?",
    "💬 What are fans saying right now?",
    "🧑‍🤝‍🧑 Who's in the current squad?",
    "📅 Full upcoming fixtures?",
]


def main():
    st.set_page_config(
        page_title="RedWire AI — Man Utd Chatbot",
        page_icon="🔴",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    # ── Header ────────────────────────────────
    st.markdown("""
    <div class="rw-header">
        <p class="rw-title">🔴 RedWire AI</p>
        <p class="rw-subtitle">Manchester United Intelligence Hub · Live Data · AI-Powered</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Session state ─────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.markdown("### 🔴 RedWire AI")
        st.caption("Your Manchester United intelligence hub")
        st.divider()

        # Man Utd ID status
        with st.spinner("Resolving Man Utd ID..."):
            team_id = get_man_utd_id()
        if team_id:
            st.success(f"✅ Connected (ID: {team_id})")
        else:
            st.warning("⚠️ Football API: team ID not resolved")

        st.divider()
        st.markdown("**📡 Live Sources**")
        st.markdown("🐦 X / Twitter")
        st.markdown("🧵 Threads")
        st.markdown("⚽ Football Data API")

        st.divider()
        st.markdown("**💬 Quick Questions**")
        for q in QUICK_QUESTIONS:
            if st.button(q, use_container_width=True, key=f"q_{q[:15]}"):
                st.session_state.pending = q

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("Built with ❤️ by a Red Devil\nPowered by Groq · RapidAPI")

    # ── Chat history ──────────────────────────
    for msg in st.session_state.messages:
        avatar = "🔴" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Input resolution ──────────────────────
    prompt = None
    if st.session_state.pending:
        prompt = st.session_state.pending
        st.session_state.pending = None
    else:
        prompt = st.chat_input("Ask me anything about Manchester United... ⚽")

    # ── Pipeline ──────────────────────────────
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🔴"):
            with st.status("🤖 RedWire is thinking...", expanded=True) as status:

                # Step 1 — Intent
                st.write("🧠 **Step 1:** Understanding your query...")
                intent_data = extract_intent(prompt)
                intent_label = intent_data.get("intent", "general")
                st.write(f"📌 Intent → **{intent_label}** | {intent_data.get('summary','')}")

                # Step 2 — Fetch
                st.write("📡 **Step 2:** Fetching live data...")
                fetched = fetch_data(intent_data, team_id)

                sources = []
                if fetched.get("x_posts"):
                    sources.append(f"X ({len(fetched['x_posts'])} posts)")
                if fetched.get("threads_posts"):
                    sources.append(f"Threads ({len(fetched['threads_posts'])} posts)")
                for k in ("live_scores","standings","fixtures","results","squad"):
                    if k in fetched:
                        sources.append(k.replace("_"," ").title())
                st.write(f"✅ Sources used: {', '.join(sources) if sources else 'None'}")

                # Step 3 — Context (RAG)
                st.write("📚 **Step 3:** Building RAG context...")
                context = build_context(fetched)

                status.update(label="✍️ Generating answer...", state="running")

            # Step 4 — Stream answer
            placeholder = st.empty()
            full_response = ""
            for chunk in stream_response(prompt, context):
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
