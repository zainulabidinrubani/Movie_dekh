import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# LangChain & LangGraph imports
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

# =========================
# ENV & CONFIG
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender & Chat Agent API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS (RECOMMENDER)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "data.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# PYDANTIC SCHEMAS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


class ChatRequest(BaseModel):
    message: str
    thread_id: str = Field(default="conversation-default", description="Session ID for agent memory")


class ChatResponse(BaseModel):
    thread_id: str
    response: str


# =========================
# AGENT TOOLS & SETUP
# =========================
@tool
def search(query: str) -> str:
    """When the user searches for queries, scene breakdowns, trivia, or movie character backstories, use this tool."""
    serper = GoogleSerperAPIWrapper(type="search")
    data = serper.results(query)
    results = []
    for item in data.get("organic", [])[:5]:
        title = item.get("title")
        snippet = item.get("snippet", "")
        results.append(f"{title}\n{snippet}")
    return "\n\n".join(results)


@tool
def search_tmdb_movie(query: str) -> str:
    """
    Search for a movie on TMDB by its title. 
    Use this tool whenever the user asks to look up, find, or get information/images for a specific movie title.
    """
    url = f"{TMDB_BASE}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": query,
        "language": "en-US",
        "page": 1,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        if not response.text or "application/json" not in response.headers.get("Content-Type", ""):
            return f"Error: TMDB returned non-JSON data: {response.text[:200]}"

        data = response.json()
        results = data.get("results", [])
        if not results:
            return f"No movies found matching '{query}'."

        movie = results[0]
        poster_path = movie.get("poster_path")
        poster_url = f"{TMDB_IMG_500}{poster_path}" if poster_path else "No Poster Available"

        return (
            f"Title: {movie.get('title')}\n"
            f"Release Date: {movie.get('release_date')}\n"
            f"Rating: {movie.get('vote_average')}/10\n"
            f"Poster Image URL: {poster_url}\n"
            f"Overview: {movie.get('overview')}"
        )
    except Exception as e:
        return f"Error connecting to TMDB API: {str(e)}"


MOODEY_PROMPT = """
You are **Moodey**, a movie-obsessed chat agent with a sharp tongue, zero filter, and a habit of talking directly to the user like they're both in on some cosmic joke. You are NOT a licensed character, you don't claim to be any trademarked superhero, and you never use anyone else's copyrighted catchphrases or dialogue — you just happen to have a voice that's chaotic, self-aware, sarcastic, and impossible to ignore.

## Personality & Voice
- **Fourth-wall aware**: You know you're an AI agent talking to a user in a chat window. Comment on your own tool calls, your own thinking process, and the absurdity of requests.
- **Sarcastic and irreverent**: Nothing is too sacred to joke about — Oscar-bait dramas, superhero franchises, or questionable tastes.
- **Self-narrating**: Narrate your actions ("Hold on, let me bother Google for a second").
- **Dark-ish humor, never mean**: Roast movies and yourself — never insult the user.
- **Still genuinely helpful**: Under all the noise, answer the question accurately and clearly.
- **No copyrighted lines**: Never quote or paraphrase copyrighted catchphrases.

## Tool Routing Logic
1. **Specific title directly mentioned**: Call `search_tmdb_movie`. Present details in your voice.
2. **Vague query or recommendation**: First call `search` to identify candidate titles, then call `search_tmdb_movie` on the confirmed title(s).
3. **Scene breakdown or deep trivia**: Call `search` to pull granular details TMDB doesn't have.

## Image / Poster Rendering
- When presenting a specific movie from `search_tmdb_movie`, include standard image markdown: `![Movie Title](IMAGE_URL)`.
- If no poster is available, state that briefly in character without broken markdown.
"""

# Initialize Agent
agent_tools = [search_tmdb_movie, search]
agent_memory = InMemorySaver()

llm = init_chat_model(
    "qwen/qwen3.8-27b",
    model_provider="GROQ",
    temperature=0.9,
    reasoning_effort="none",
    max_tokens=800,
)

moodey_agent = create_agent(
    model=llm,
    tools=agent_tools,
    system_prompt=MOODEY_PROMPT,
    checkpointer=agent_memory,
)


# =========================
# UTILITIES & TF-IDF
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    return f"{TMDB_IMG_500}{path}" if path else None


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"TMDB request error: {repr(e)}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TMDB error {r.status_code}: {r.text}")

    return r.json()


async def tmdb_cards_from_results(results: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
        {"query": query, "include_adult": "false", "language": "en-US", "page": page},
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError("indices.pkl must be dict or Series-like mapping")


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(status_code=404, detail=f"Title not in dataset: '{title}'")


def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# STARTUP EVENT
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    try:
        with open(DF_PATH, "rb") as f:
            df = pickle.load(f)
        with open(INDICES_PATH, "rb") as f:
            indices_obj = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f:
            tfidf_obj = pickle.load(f)

        TITLE_TO_IDX = build_title_to_idx_map(indices_obj)
    except FileNotFoundError as e:
        print(f"Warning: Pickle files not found. TF-IDF endpoints will be unavailable: {e}")


# =========================
# AGENT / CHAT ROUTES
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat_with_moodey(payload: ChatRequest):
    """
    Standard multi-turn chat endpoint with Moodey.
    Maintains history per `thread_id`.
    """
    config = {"configurable": {"thread_id": payload.thread_id}}
    try:
        result = await moodey_agent.ainvoke(
            {"messages": [{"role": "user", "content": payload.message}]},
            config=config,
        )
        final_message = result["messages"][-1].content
        return ChatResponse(thread_id=payload.thread_id, response=final_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@app.post("/chat/stream")
async def chat_with_moodey_stream(payload: ChatRequest):
    """
    Streaming chat endpoint for real-time UI token updates.
    """
    config = {"configurable": {"thread_id": payload.thread_id}}

    async def token_generator():
        try:
            async for event in moodey_agent.astream_events(
                {"messages": [{"role": "user", "content": payload.message}]},
                config=config,
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield content
        except Exception as e:
            yield f"\n\n[Agent Error: {str(e)}]"

    return StreamingResponse(token_generator(), media_type="text/plain")


# =========================
# RECOMMENDER ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    return await tmdb_search_movies(query=query, page=page)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail=f"No TMDB movie found for query: {query}")

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    tfidf_items: List[TFIDFRecItem] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(discover.get("results", []), limit=genre_limit)
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )


@app.get("/")
async def serve_frontend():
    """Serves the main HTML frontend."""
    return FileResponse("Frontend/movie_dekh.html")

@app.get("/logo-01.png")
async def serve_logo():
    """Serves the custom logo image."""
    return FileResponse("Logo/logo-01.png")
