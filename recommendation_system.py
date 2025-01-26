import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

# Load datasets
movies = pd.read_csv("movies.csv")  # Movie metadata
ratings = pd.read_csv("ratings.csv")  # User ratings

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

# Display a sample of the merged data
print(data.head())

# -----------------------------
# 1. Content-Based Filtering
# -----------------------------

def content_based_recommendations(movie_title, top_n=10):
    """
    Recommend movies similar to the given movie using genres and content-based filtering.
    """
    # Create a TF-IDF Vectorizer to process genres
    tfidf = TfidfVectorizer(stop_words="english")
    movies["genres"] = movies["genres"].fillna("")  # Handle missing genres
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index of the given movie
    idx = movies[movies["title"] == movie_title].index[0]

    # Get similarity scores for the movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top-n most similar movies
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    # Return movie titles
    return movies["title"].iloc[movie_indices]


# -----------------------------
# 2. Collaborative Filtering
# -----------------------------

def collaborative_filtering_recommendations(user_id, top_n=10):
    """
    Recommend movies for a user based on collaborative filtering using user ratings.
    """
    # Pivot the dataset to create a user-item matrix
    user_item_matrix = data.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # Create a sparse matrix
    user_item_sparse = csr_matrix(user_item_matrix.values)

    # Compute cosine similarity between users
    user_sim = cosine_similarity(user_item_sparse)

    # Get the index of the given user
    user_idx = user_id - 1  # Adjust for zero-indexing
    user_scores = list(enumerate(user_sim[user_idx]))
    user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)

    # Find movies rated highly by similar users
    similar_users = [user[0] for user in user_scores[1:top_n + 1]]
    similar_users_ratings = user_item_matrix.iloc[similar_users].mean(axis=0)

    # Recommend the top-n unrated movies for the user
    user_ratings = user_item_matrix.iloc[user_idx]
    unrated_movies = user_ratings[user_ratings == 0].index
    recommendations = similar_users_ratings.loc[unrated_movies].sort_values(ascending=False).head(top_n)

    # Return recommended movie titles
    return movies[movies["movieId"].isin(recommendations.index)]["title"].tolist()


# -----------------------------
# Showcase and Testing
# -----------------------------

if __name__ == "__main__":
    # Example 1: Content-Based Recommendations
    movie = "Toy Story (1995)"
    print(f"Movies similar to '{movie}':")
    print(content_based_recommendations(movie))

    # Example 2: Collaborative Filtering Recommendations
    user = 1
    print(f"\nMovies recommended for user {user}:")
    print(collaborative_filtering_recommendations(user))
