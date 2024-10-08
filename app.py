import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
movies = pd.read_csv('movies.dat', sep='::', names=['movieId', 'title', 'genres'], engine='python', encoding='ISO-8859-1')
ratings = pd.read_csv('ratings.dat', sep='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python', encoding='ISO-8859-1')

# Drop any missing values
movies.dropna(inplace=True)
ratings.dropna(inplace=True)

# Load the collaborative filtering model
best_model = joblib.load('collaborative_model.pkl')

# Content-Based Filtering with optimized TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].str.replace('|', ' ')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_based_recommendations(movie_title, num_recommendations=10):
    try:
        movie_idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(content_sim[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices]
    except IndexError:
        st.error("Movie title not found in the dataset. Please enter a valid movie title.")
        return []

def hybrid_recommendations(user_id, movie_title, num_recommendations=10, weight_collab=0.5, weight_content=0.5):
    try:
        movie_id = movies[movies['title'] == movie_title]['movieId'].iloc[0]
    except IndexError:
        st.error("Movie title not found in the dataset. Please enter a valid movie title.")
        return []
    
    collab_pred = best_model.predict(user_id, movie_id).est
    content_based_rec = content_based_recommendations(movie_title, num_recommendations)
    hybrid_scores = {}
    for title in content_based_rec:
        movie_id = movies[movies['title'] == title]['movieId'].iloc[0]
        collab_score = best_model.predict(user_id, movie_id).est
        hybrid_score = weight_collab * collab_score + weight_content * collab_pred
        hybrid_scores[title] = hybrid_score
    sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return [(rec, score) for rec, score in sorted_recommendations[:num_recommendations]]

def plot_recommendations(recommendations):
    if recommendations:
        movie_titles = [rec[0] for rec in recommendations]
        scores = [rec[1] for rec in recommendations]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=scores, y=movie_titles, palette='viridis')
        plt.xlabel('Hybrid Recommendation Score')
        plt.title('Top Hybrid Recommendations')
        st.pyplot(plt)
    else:
        st.error("No recommendations to display.")

# Streamlit app
st.set_page_config(page_title='Movie Recommendation System', layout='wide')

st.title('Movie Recommendation System')

st.sidebar.header('User Input Features')
user_id = st.sidebar.slider('Select User ID:', min_value=1, max_value=100, value=1)
movie_title = st.sidebar.text_input('Enter a Movie Title:')

if st.sidebar.button('Get Recommendations'):
    if movie_title:
        with st.spinner('Generating recommendations...'):
            recommended_movies = hybrid_recommendations(user_id, movie_title)
        if recommended_movies:
            st.success('Recommendations generated!')

            st.subheader('Top Recommendations:')
            for idx, (rec, score) in enumerate(recommended_movies):
                st.write(f"{idx+1}. {rec} (Score: {score:.2f})")

            st.subheader('Visualization of Recommendations:')
            plot_recommendations(recommended_movies)
    else:
        st.error('Please enter a movie title.')

st.sidebar.subheader('About')
st.sidebar.info("This app provides movie recommendations based on a hybrid model combining collaborative filtering and content-based filtering. Adjust the user ID and enter a movie title to get personalized recommendations.")
