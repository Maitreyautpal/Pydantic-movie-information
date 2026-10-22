# Pydantic Movie Information

An AI-powered movie information extraction app built using Streamlit, LangChain, Groq, and Pydantic.

## Features
- Extracts structured movie information from a paragraph
- Identifies title, release year, genre, director, cast, rating, and summary
- Displays structured JSON
- Allows downloading the extracted data as JSON

## Run Locally
1. Install dependencies:
   `pip install -r requirements.txt`
2. Set `GROQ_API_KEY` in your `.env` file.
3. Run:
   `streamlit run UIpydantic.py`