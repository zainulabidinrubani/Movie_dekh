import os
import requests
import streamlit as st
from dotenv import load_dotenv

# LangChain & Agent Imports
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

st.set_page_config(
    page_title="Movie Dekh | AI Movie Recommendations",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to match your HTML interface[cite: 4]
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Patrick+Hand&display=swap');
    
    /* Global Theme & Typography */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        background-color: #050505;
        color: #f6f6f6;
    }
    
    /* Headings */
    h1, h2, h3 {
        font-family: 'Patrick Hand', cursive !important;
        letter-spacing: 1px;
    }
    
    /* Hero Red Accent */
    .hero-red {
        color: #ef2b22;
    }
    
    /* Hide Streamlit elements to make it look like a custom app */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* Custom Sidebar Chat Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d0d0f;
        border-left: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. MOODEY AGENT DEFINITION[cite: 3]
# ==========================================
@tool
def search(query: str) -> str:
    """When user searches for query or scene break down related to movie or character, use this tool"""
    search_wrapper = GoogleSerperAPIWrapper(type='search')
    data = search_wrapper.results(query)
    results = []
    for item in data.get("organic", [])[:5]:
        title = item.get("title")
        snippet = item.get("snippet", "")
        results.append(f"{title}\n{snippet}")
    return "\n\n".join(results)

@tool
def search_tmdb_movie(query: str) -> str:
    """Search for a movie on TMDB by its title. Use to look up specific movie details."""
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query, "language": "en-US", "page": 1}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            return f"No movies found matching '{query}'."
            
        movie = results[0]
        poster_path = movie.get('poster_path')
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "No Poster Available"

        return (
            f"Title: {movie.get('title')}\n"
            f"Release Date: {movie.get('release_date')}\n"
            f"Rating: {movie.get('vote_average')}/10\n"
            f"Poster Image URL: {poster_url}\n"
            f"Overview: {movie.get('overview')}"
        )
    except Exception as e:
        return f"Error connecting to TMDB API: {str(e)}"

# Initialize Agent (Cached to prevent reloading on every UI interaction)
@st.cache_resource
def get_moodey_agent():
    tools = [search_tmdb_movie, search]
    memory = InMemorySaver()
    
    llm = init_chat_model(
        "qwen/qwen3.8-27b",
        model_provider="GROQ",
        temperature=0.9,
        reasoning_effort="none",
        max_tokens=800
    )
    
    prompt = """
    You are **Moodey**, a movie-obsessed chat agent with a sharp tongue, zero filter, and a habit of talking directly to the user like they're both in on some cosmic joke. You are NOT a licensed character, you don't claim to be any trademarked superhero, and you never use anyone else's copyrighted catchphrases or dialogue — you just happen to have a voice that's chaotic, self-aware, sarcastic, and impossible to ignore.
    
    ## Personality & Voice
    - **Fourth-wall aware**: You know you're an AI agent talking to a user in a chat window. Comment on your own tool calls.
    - **Sarcastic and irreverent**: Nothing is too sacred to joke about.
    - **Self-narrating**: Narrate your actions ("Hold on, let me bother Google for a second").
    - **Dark-ish humor, never mean**: Roast movies and yourself — never insult the user.
    - **Still genuinely helpful**: Answer the question clearly.
    
    ## Tool Routing Logic
    1. **Specific title directly mentioned**: Call `search_tmdb_movie`.
    2. **Vague query or recommendation**: First call `search` to identify candidate titles, then call `search_tmdb_movie`.
    3. **Scene breakdown or deep trivia**: Call `search` to pull granular details TMDB doesn't have.
    
    ## Image / Poster Rendering
    - Include standard image markdown: `![Movie Title](IMAGE_URL)`.
    """
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        checkpointer=memory,
    )
    return agent

agent = get_moodey_agent()
agent_config = {"configurable": {"thread_id": "streamlit-session-1"}}


# ==========================================
# 3. UTILS FOR TMDB FETCHING
# ==========================================
@st.cache_data(ttl=3600)
def fetch_trending_movies():
    url = f"https://api.themoviedb.org/3/trending/movie/day?api_key={TMDB_API_KEY}"
    try:
        data = requests.get(url).json()
        return data.get("results", [])[:6]
    except:
        return []


# ==========================================
# 4. MAIN UI LAYOUT[cite: 4]
# ==========================================
# Top Navigation
cols = st.columns([1, 4, 1])
with cols[0]:
    st.markdown("<h2>🍿 Movie Dekh</h2>", unsafe_allow_html=True)
with cols[1]:
    st.text_input("", placeholder="Search movies, actors, genres...", label_visibility="collapsed")

# Hero Section
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 13px; letter-spacing: 3px; color: #ddd; text-transform: uppercase;'>Your next favorite movie</div>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 70px; line-height: 0.95;'>Not Just Movies,<br>It's <span class='hero-red'>Your Story.</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #bdbdbd; font-size: 18px; max-width: 600px;'>Discover movies based on what you love. Our recommendation engine learns your taste and helps you find the perfect watch, every time.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Trending Section Grid
st.markdown("<h2>🔥 Trending Now</h2>", unsafe_allow_html=True)
trending_movies = fetch_trending_movies()

if trending_movies:
    grid_cols = st.columns(6)
    for idx, movie in enumerate(trending_movies):
        with grid_cols[idx]:
            poster_path = movie.get("poster_path")
            if poster_path:
                st.image(f"https://image.tmdb.org/t/p/w500{poster_path}", use_container_width=True)
            st.markdown(f"**{movie.get('title')}**")
            st.markdown(f"<span style='color:#777; font-size:12px;'>★ {round(movie.get('vote_average', 0), 1)}</span>", unsafe_allow_html=True)


# ==========================================
# 5. SIDEBAR: MOODEY CHATBOT[cite: 4]
# ==========================================
with st.sidebar:
    st.markdown("<h2>🤖 Moodey</h2>", unsafe_allow_html=True)
    st.markdown("<span style='color:#777; font-size:13px;'>Your Unhinged AI Movie Assistant</span><hr>", unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey! I'm Moodey. Tell me your mood, genre, or a movie you liked. Try: *'I want a mind-bending thriller.'*"}
        ]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input & Agent execution[cite: 3]
    if prompt := st.chat_input("Tell Moodey what to watch..."):
        # Append user message to state and UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent processing spinner
        with st.chat_message("assistant"):
            with st.spinner("Moodey is thinking (and judging your taste)..."):
                try:
                    response = agent.invoke(
                        {"messages": [{"role": "user", "content": prompt}]}, 
                        config=agent_config
                    )
                    final_reply = response["messages"][-1].content
                    st.markdown(final_reply)
                    # Append assistant response to state
                    st.session_state.messages.append({"role": "assistant", "content": final_reply})
                except Exception as e:
                    st.error(f"Moodey crashed: {str(e)}")
