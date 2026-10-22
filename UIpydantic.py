
import os
import json
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MovieMind | Movie Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #312e81);
        color: white;
        margin-bottom: 25px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }
    .hero p {
        color: #e0e7ff;
        font-size: 16px;
    }
    .detail-card {
        padding: 16px;
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 12px;
        min-height: 100px;
        margin-bottom: 12px;
    }
    .detail-label {
        font-size: 13px;
        opacity: 0.75;
        margin-bottom: 7px;
    }
    .detail-value {
        font-size: 18px;
        font-weight: 600;
        overflow-wrap: anywhere;
    }
    div.stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- PYDANTIC SCHEMA ----------------
class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Extract movie information from the paragraph provided by the user.

        Use only information supported by the paragraph or information
        you can confidently identify. Do not invent facts.
        If information is unavailable, use null for optional fields
        and an empty list for list fields.

        {format_instructions}
        """
    ),
    ("human", "{paragraph}")
])


# ---------------- HELPERS ----------------
def detail_card(label, value):
    if value is None or value == "":
        value = "Not specified"

    st.markdown(
        f"""
        <div class="detail-card">
            <div class="detail-label">{label}</div>
            <div class="detail-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_model():
    api_key = os.getenv("GROQ_API_KEY")

    # Optional support for Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. Please configure it in your "
            ".env file or Streamlit secrets."
        )

    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0
    )


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🎬 MovieMind</h1>
    <p>AI-powered movie information extraction using LangChain, Groq and Pydantic.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("🔍 Analyze a Movie")
st.write(
    "Paste a paragraph about a movie, and AI will convert the available "
    "information into structured movie details."
)

# ---------------- INPUT ----------------
with st.container(border=True):
    paragraph = st.text_area(
        "Movie Paragraph",
        height=180,
        placeholder=(
            "Example: Inception is a science fiction film directed by "
            "Christopher Nolan and released in 2010..."
        ),
        help="Enter a movie description or summary in English."
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        analyze = st.button(
            "✨ Extract Details",
            type="primary",
            use_container_width=True
        )

    with col2:
        clear = st.button(
            "Clear Results",
            use_container_width=True
        )

if clear:
    st.session_state.pop("movie_result", None)
    st.rerun()

# ---------------- ANALYSIS ----------------
if analyze:
    if not paragraph.strip():
        st.warning("Please enter a movie paragraph before proceeding.")

    else:
        try:
            with st.spinner("🎬 Extracting movie details using AI..."):
                model = get_model()

                final_prompt = prompt.invoke({
                    "paragraph": paragraph,
                    "format_instructions": parser.get_format_instructions()
                })

                response = model.invoke(final_prompt)
                movie_data = parser.parse(response.content)

                st.session_state["movie_result"] = movie_data.model_dump()

        except Exception as e:
            st.error("Unable to complete the movie analysis.")
            st.caption(f"Error: {str(e)}")

# ---------------- RESULTS ----------------
result = st.session_state.get("movie_result")

if result:
    movie = Movie.model_validate(result)

    st.divider()
    st.subheader("📽️ Movie Analysis Results")

    st.markdown(f"## {movie.title}")

    # Quick information cards
    c1, c2, c3 = st.columns(3)

    with c1:
        detail_card(
            "📅 Release Year",
            str(movie.release_year) if movie.release_year is not None
            else "Not specified"
        )

    with c2:
        detail_card(
            "🎭 Genre",
            ", ".join(movie.genre) if movie.genre else "Not specified"
        )

    with c3:
        detail_card(
            "⭐ Rating",
            str(movie.rating) if movie.rating is not None
            else "Not specified"
        )

    # Director and cast
    c4, c5 = st.columns(2)

    with c4:
        st.markdown("### 🎥 Director")
        st.write(movie.director or "Not specified")

    with c5:
        st.markdown("### 👥 Cast")
        if movie.cast:
            for actor in movie.cast:
                st.markdown(f"- {actor}")
        else:
            st.write("Not specified")

    # Summary
    st.markdown("### 📝 Movie Summary")
    st.write(movie.summary or "Not specified")

    # JSON output
    st.divider()

    with st.expander("🧩 View Structured JSON"):
        st.json(result)

        st.download_button(
            label="⬇️ Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="movie_details.json",
            mime="application/json"
        )

else:
    st.info("Submit a movie paragraph to view the extracted results here.")

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with Streamlit • LangChain • Groq • Pydantic")
