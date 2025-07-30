import requests
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os


script_dir = os.path.dirname(__file__)


movies_dict_file_path = os.path.join(script_dir, 'movie_dict.pkl')
similarity_file_path = os.path.join(script_dir, 'similarity.pkl')

def fetch_poster(movie_id):
    try:
        response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US')
        response.raise_for_status()
        data = response.json()

        if data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster" # Placeholder image
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching poster for movie ID {movie_id}: {e}")
        return "https://via.placeholder.com/500x750?text=Error+Loading+Poster"
    except KeyError:
        st.warning(f"Poster path not found for movie ID {movie_id} in API response. Defaulting to placeholder.")
        return "https://via.placeholder.com/500x750?text=No+Poster+Path"
    except Exception as e:
        st.error(f"An unexpected error occurred in fetch_poster for movie ID {movie_id}: {e}")
        return "https://via.placeholder.com/500x750?text=Error"


def recommendations(movie_title, movies_df, similarity_matrix):
    if 'title' not in movies_df.columns:
        st.error("Error: 'title' column not found in movies DataFrame.")
        return [], []

    if movie_title not in movies_df['title'].values:
        return ["Movie not found in the database. Please select another."], []

    movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    distances = similarity_matrix[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_posters = []
    recommended_movies = []
    for i in movies_list:
        try:

            movie_id = movies_df.iloc[i[0]]['movie_id_y']
            recommended_movies.append(movies_df.iloc[i[0]]['title'])
            recommended_posters.append(fetch_poster(movie_id))
        except KeyError as e:
            st.error(f"Column '{e}' not found in your movies DataFrame. Please check your 'movie_dict.pkl' file for column names.")
            return [], []
        except IndexError:
            st.warning(f"Index error for movie list. Skipping recommendation {i[0]}.")
            continue

    return recommended_movies, recommended_posters

try:
    movies_dict = pickle.load(open(movies_dict_file_path, 'rb'))
    movies = pd.DataFrame(movies_dict)

    similarity = pickle.load(open(similarity_file_path, 'rb'))

    st.title("Movie Recommendation System")

    if 'title' in movies.columns:
        option = st.selectbox(
            "Select a movie to get recommendations:",
            movies['title'].values
        )
    else:
        st.error("Cannot load movie titles for selection. 'title' column missing from data.")
        option = None

    if st.button("Recommend") and option is not None:
        recommended_movie_names, recommended_poster_urls = recommendations(option, movies, similarity)

        if recommended_movie_names and recommended_poster_urls:
            st.subheader("Recommended Movies:")

            cols = st.columns(5)

            for i in range(min(len(recommended_movie_names), 5)):
                with cols[i]:
                    st.text(recommended_movie_names[i])
                    st.image(recommended_poster_urls[i])
        else:
            st.warning("Could not find recommendations for the selected movie or data is incomplete.")
            if recommended_movie_names and "Movie not found" in recommended_movie_names:
                st.info("The selected movie might not be in the recommendation engine's database.")

except FileNotFoundError as e:
    st.error(f"Error: A required file was not found. {e}")
    st.info(f"Please ensure '{os.path.basename(e.filename)}' is in the same directory as 'app.py' or that the path is correct.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.info(f"Please check your data files and code for any inconsistencies. Specific error: {e}")