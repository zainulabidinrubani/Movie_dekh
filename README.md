Movie Dekh is your own personal AI-powered movie buddy app designed to cure the endless "what should we watch tonight?" scrolling problem.

Instead of just showing a static list of trending films, it combines smart machine learning models with an interactive AI assistant named **Moodey**. You can actually chat with Moodey, describe the exact vibe, genre, or mood you're looking for, and get tailored recommendations on the fly.

Behind the scenes, it pulls live posters, ratings, and synopses directly from the TMDB database and uses the Serper API for real-time web searches. It's built with a high-performance **FastAPI** backend, managed using lightning-fast `uv` packaging, and successfully deployed live to the cloud on Render with secure environment variables. It’s essentially a blend of a classic recommendation engine and a modern conversational AI built right into a clean web interface!

**Project Structure**

* **`frontend/`**: Holds all your HTML templates, stylesheets, and frontend code that build the user interface.
* **`logo/`**: Contains the visual branding assets, icons, and logo graphics for the application.
* **`main.py`**: The core FastAPI backend script that runs the server, handles web routes, loads your `.pkl` machine learning models, and powers the Moodey assistant.
* **`pyproject.toml` & `uv.lock**`: The configuration and lock files used by `uv` to manage and install project dependencies instantly.
* **`README.md`**: The documentation file explaining what the project does and how to run it.

**How to Use Your Live Link**

Once Render finishes deploying your project, it provides a public web URL ending in `https://movie-dekh.onrender.com/`.

* **Accessing the App:** Click your Render service URL to open the website in your browser. If the app hasn't been used in a while, the very first load might take 30 to 50 seconds to wake up from its sleep mode.
* **Chatting with Moodey:** Once the page loads, type out your movie preferences or describe the kind of mood or genre you want to watch in the chat interface.
* **Exploring Results:** The app will process your request, pull live posters, ratings, and synopses from the TMDB database, and display your personalized recommendations directly on the screen.
