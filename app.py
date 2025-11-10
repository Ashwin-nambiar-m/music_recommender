import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(
    page_title="🎧 AI Music Recommender",
    page_icon="🎵",
    layout="wide",
)

# Custom CSS styling for Spotify-like dark theme
st.markdown("""
  <style>
    /* Background - Professional dark gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        color: #e0e0e0;
        animation: gradientShift 15s ease infinite;
    }

    @keyframes gradientShift {
        0% {
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        }
        50% {
            background: linear-gradient(135deg, #16162a 0%, #1f1f3a 100%);
        }
        100% {
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        }
    }

    /* Main titles */
    h1, h2, h3 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        animation: titleFade 1s ease-in;
    }

    @keyframes titleFade {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Buttons - Dual colored gradient */
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.4s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }

    div.stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transition: left 0.4s ease;
        z-index: -1;
    }

    div.stButton > button:hover::before {
        left: 0;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Track cards - Professional dark style */
    .track-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #252538 100%);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: cardSlideIn 0.6s ease-out;
    }

    @keyframes cardSlideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .track-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.15), transparent);
        transition: left 0.5s ease;
    }

    .track-card:hover::before {
        left: 100%;
    }

    .track-card:hover {
        background: linear-gradient(145deg, #252538 0%, #2d2d44 100%);
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.4);
    }

    /* Sidebar - Dark professional */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a14 0%, #12121f 100%) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }

    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Input fields */
    input, textarea, select {
        background-color: #1e1e2e !important;
        color: #e0e0e0 !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }

    input:focus, textarea:focus, select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        background-color: #252538 !important;
    }

    /* Metric containers */
    [data-testid="stMetricValue"] {
        color: #667eea !important;
        font-weight: 700;
        animation: numberPulse 2s ease-in-out infinite;
    }

    @keyframes numberPulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.02);
        }
    }

    /* Text elements */
    p, span, div {
        color: #e0e0e0;
    }

    /* Loading animation */
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Data frames and tables */
    [data-testid="stDataFrame"] {
        background-color: #1e1e2e !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #1e1e2e;
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
    }

    /* Tabs */
    [data-baseweb="tab"] {
        color: #e0e0e0 !important;
        background-color: transparent !important;
    }

    [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.1) !important;
    }

    [aria-selected="true"] {
        border-bottom: 2px solid #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data(path="spotify_tracks.csv"):
    df = pd.read_csv(path)
    numeric_cols = [
        "acousticness","danceability","duration_ms","energy","instrumentalness",
        "key","liveness","loudness","mode","speechiness","tempo","time_signature",
        "valence","popularity","year"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df = df.dropna(subset=["track_name"])
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
    df["title_display"] = df["track_name"] + " — " + df["artist_name"]
    return df, numeric_cols

# ---------------------- MODEL SETUP ----------------------
@st.cache_resource
def build_model(features):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(X)
    return model, scaler

def recommend(df, model, scaler, feature_cols, base_idx, k=10):
    vec = df.loc[base_idx, feature_cols].values.reshape(1, -1)
    vec_scaled = scaler.transform(vec)
    distances, indices = model.kneighbors(vec_scaled, n_neighbors=k+1)
    return indices[0][1:]

# ---------------------- DISPLAY TRACK ----------------------
def show_track(row):
    col1, col2 = st.columns([1, 3])
    with col1:
        if isinstance(row.get("artwork_url"), str) and row["artwork_url"].startswith("http"):
            st.image(row["artwork_url"], use_container_width=True)
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/727/727245.png", width=150)
    with col2:
        st.markdown(f"<div class='track-card'>"
                    f"<h3>{row['track_name']}</h3>"
                    f"<p><b>Artist:</b> {row['artist_name']}</p>"
                    f"<p><b>Album:</b> {row.get('album_name', 'N/A')}</p>"
                    f"<p><b>Year:</b> {row.get('year', '—')}</p>"
                    f"</div>", unsafe_allow_html=True)
        if isinstance(row.get("track_url"), str) and row["track_url"].startswith("http"):
            st.link_button("🎵 Play on Spotify", row["track_url"], type="primary")

# ---------------------- MAIN APP ----------------------
def main():
    st.title("🎧 AI Music Recommendation System")
    st.caption("Discover new songs that match your vibe — Powered by AI")

    # Load dataset
    df, features = load_data("spotify_tracks.csv")
    model, scaler = build_model(df[features].values)

    st.sidebar.header("⚙️ Controls")
    mode = st.sidebar.radio("Recommendation Mode", ["By Song", "By Artist", "By Mood"])
    st.sidebar.markdown("---")

    # ---------------------- BY SONG ----------------------
    if mode == "By Song":
        st.subheader("🎵 Search by Song Name or Artist")
        search = st.text_input("Enter a song or artist:")
        results = df[df["title_display"].str.contains(search, case=False, na=False)] if search else df.head(15)
        selected = st.selectbox("Select a song", results["title_display"].values)
        base_idx = results.index[results["title_display"] == selected][0]

        st.markdown("### 🔘 Selected Song")
        show_track(df.loc[base_idx])

        st.markdown("### 🔮 Recommended Songs")
        rec_indices = recommend(df, model, scaler, features, base_idx, k=8)
        for idx in rec_indices:
            show_track(df.loc[idx])

    # ---------------------- BY ARTIST ----------------------
    elif mode == "By Artist":
        st.subheader("🎤 Choose an Artist")
        artist = st.selectbox("Select Artist", sorted(df["artist_name"].unique()))
        artist_tracks = df[df["artist_name"] == artist]
        mean_vec = artist_tracks[features].mean().values.reshape(1, -1)
        mean_scaled = scaler.transform(mean_vec)
        distances, indices = model.kneighbors(mean_scaled, n_neighbors=9)

        st.markdown(f"### 🔮 Because you like {artist}")
        for idx in indices[0][1:]:
            show_track(df.loc[idx])

    # ---------------------- BY MOOD ----------------------
    else:
        st.subheader("🌈 Set Your Mood")
        st.write("Adjust the sliders to find tracks that fit your energy and vibe.")
        with st.sidebar:
            dance = st.slider("💃 Danceability", 0.0, 1.0, 0.5)
            energy = st.slider("⚡ Energy", 0.0, 1.0, 0.5)
            valence = st.slider("😊 Positivity (Valence)", 0.0, 1.0, 0.5)
            tempo = st.slider("🎚️ Tempo (BPM)", 50.0, 200.0, 120.0)

        feature_subset = [f for f in ["danceability", "energy", "valence", "tempo"] if f in df.columns]
        X_sub = df[feature_subset].values
        model_sub, scaler_sub = build_model(X_sub)
        user_vec = np.array([[dance, energy, valence, tempo]])
        dists, inds = model_sub.kneighbors(scaler_sub.transform(user_vec), n_neighbors=8)

        st.markdown("### ✨ Songs That Match Your Mood")
        for idx in inds[0]:
            show_track(df.loc[idx])

    st.markdown("---")
    st.caption("Built with ❤️ | Designed by Ashwin 🎧")

# ---------------------- RUN ----------------------
if __name__ == "__main__":
    main() 
