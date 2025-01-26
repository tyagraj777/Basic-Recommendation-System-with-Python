#  Basic-Recommendation-System-with-Python
Basic recommendation system in Python. The example demonstrates collaborative filtering and content-based filtering for a movie recommendation system using Pandas and scikit-learn.


# Folder Structure

![image](https://github.com/user-attachments/assets/71cfbbf7-6f54-4dde-9a3a-0c4c88e5ac9d)


# Datasets

You can use the MovieLens dataset as an example. or Download the datasets from MovieLens:

movies.csv:

Contains movie metadata (movieId, title, genres).

ratings.csv:

Contains user ratings (userId, movieId, rating, timestamp).



1. Install dependencies:
    ```bash
    pandas==1.3.5
    scikit-learn==1.1.3


# How It Works

1. Content-Based Filtering:
   
   Uses movie genres to calculate similarity.

   TF-IDF Vectorizer extracts important terms (genres) and computes cosine similarity to recommend similar movies.

2. Collaborative Filtering:

   Creates a user-item matrix of ratings.

   Computes cosine similarity between users.

   Recommends movies highly rated by similar users.



# Running the System

1. Install dependencies:
   ```bash
   pip install -r requirements.txt


2. Run the script:
  ```bash
  python recommendation_system.py



