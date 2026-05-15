"""
RedWire AI — Manchester United Football Chatbot
Pipeline: User Query → Agent (Intent Extraction) → API Fetch → RAG → LLM Response
"""

import streamlit as st
import requests
import json
import os
import base64
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images.jpeg")

# ─────────────────────────────────────────────
# CONFIG — st.secrets (Streamlit Cloud) with .env fallback
# ─────────────────────────────────────────────
def _secret(key: str, default: str = None) -> str:
    """Read from st.secrets first, then os.getenv."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


def _logo_data_uri(path: str) -> str | None:
    """Encode local logo for reliable HTML rendering in Streamlit."""
    try:
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None

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
        if not isinstance(t, dict):
            continue
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

    :root {
        --rw-bg: #0b0f14;
        --rw-panel: #111827;
        --rw-panel-2: #0f172a;
        --rw-border: #223046;
        --rw-text: #e6e8ee;
        --rw-muted: #a3adc2;
        --rw-accent: #ef4444;
        --rw-accent-2: #f59e0b;
        --rw-success: #22c55e;
        --rw-warning: #fbbf24;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--rw-text);
    }

    .stApp {
        background: radial-gradient(1200px 700px at 20% -10%, rgba(239,68,68,0.18), transparent 60%),
                    radial-gradient(1200px 700px at 120% 10%, rgba(245,158,11,0.15), transparent 55%),
                    var(--rw-bg);
        min-height: 100vh;
    }

    /* Main container width */
    section.main > div {
        max-width: 1100px;
        margin: 0 auto;
    }

    /* Header */
    .rw-header {
        text-align: center;
        padding: 1.8rem 0 0.6rem;
    }
    .rw-title-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.7rem;
    }
    .rw-logo {
        width: 52px;
        height: 52px;
        border-radius: 10px;
        object-fit: cover;
        border: 1px solid var(--rw-border);
        box-shadow: 0 6px 16px rgba(2, 6, 23, 0.45);
    }
    .rw-logo-sidebar {
        width: 34px;
        height: 34px;
        border-radius: 8px;
    }
    .rw-logo-fallback {
        font-size: 1.8rem;
        line-height: 1;
    }
    .rw-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 6px 24px rgba(239,68,68,0.25);
        margin: 0;
    }
    .rw-subtitle {
        color: var(--rw-muted);
        font-size: 0.95rem;
        margin-top: 0.35rem;
        letter-spacing: 0.3px;
    }

    /* Status + cards */
    [data-testid="stStatus"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.9), rgba(2,6,23,0.9)) !important;
        border: 1px solid var(--rw-border) !important;
        border-radius: 12px !important;
        color: var(--rw-text) !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid var(--rw-border);
        border-radius: 14px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 6px 18px rgba(2, 6, 23, 0.35);
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: #0b1220 !important;
        border: 1px solid var(--rw-border) !important;
        border-radius: 12px !important;
        color: var(--rw-text) !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--rw-accent) !important;
        box-shadow: 0 0 0 3px rgba(239,68,68,0.2) !important;
    }

    /* Buttons */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #0f172a, #111827);
        border: 1px solid var(--rw-border);
        color: var(--rw-text);
        border-radius: 10px;
        font-size: 0.82rem;
        padding: 0.55rem 0.8rem;
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: var(--rw-accent);
        color: #fff;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(239,68,68,0.25);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220, #0b0f14);
        border-right: 1px solid var(--rw-border);
    }
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: var(--rw-text) !important;
    }
    .rw-sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 0.25rem;
    }
    .rw-sidebar-title {
        margin: 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--rw-text);
    }
    .rw-sidebar-subtitle {
        margin: 0.1rem 0 0;
        color: var(--rw-muted);
        font-size: 0.82rem;
        line-height: 1.25;
    }

    /* Status colors */
    .stSuccess {
        background: rgba(34,197,94,0.15) !important;
        border: 1px solid rgba(34,197,94,0.35) !important;
        color: #dcfce7 !important;
    }
    .stWarning {
        background: rgba(251,191,36,0.16) !important;
        border: 1px solid rgba(251,191,36,0.4) !important;
        color: #fef3c7 !important;
    }

    /* Divider */
    hr { border-color: var(--rw-border) !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0b0f14; }
    ::-webkit-scrollbar-thumb { background: #2b3647; border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: #3b475b; }
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
    logo_uri = _logo_data_uri(LOGO_PATH)
    header_logo_html = (
        f'<img src="{logo_uri}" alt="RedWire logo" class="rw-logo" />'
        if logo_uri
        else '<span class="rw-logo-fallback">🔴</span>'
    )
    sidebar_logo_html = (
        f'<img src="{logo_uri}" alt="RedWire logo" class="rw-logo rw-logo-sidebar" />'
        if logo_uri
        else '<span class="rw-logo-fallback">🔴</span>'
    )

    st.set_page_config(
        page_title="RedWire AI — Man Utd Chatbot",
        page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🔴",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    # ── Header ────────────────────────────────
    st.markdown(f"""
    <div class="rw-header">
        <div class="rw-title-row">
            {header_logo_html}
            <p class="rw-title">RedWire AI</p>
        </div>
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
        st.markdown(f"""
        <div class="rw-sidebar-brand">
            {sidebar_logo_html}
            <div>
                <p class="rw-sidebar-title">RedWire AI</p>
                <p class="rw-sidebar-subtitle">Your Manchester United intelligence hub</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
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
