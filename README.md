# AI Music Recommender (Streamlit)

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

Place your dataset CSV in the same folder (default file name expected by the app is `spotify_tracks.csv`).

## How it works
- Uses content-based filtering with cosine similarity on Spotify audio features
- Three modes: By Song, By Artist, By Mood/Features
- Filters for language and year
- Caches data and model for fast interactions
