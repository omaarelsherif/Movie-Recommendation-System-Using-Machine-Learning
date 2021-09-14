### Movie Recommendation System ###
"""
Description:
              Recommendation systems are among the most popular applications of data science.
              They are used to predict the Rating or Preference that a user would give to an item.
              Almost every major company has applied them in some form or the other, Amazon uses it
              to suggest products to customers, YouTube uses it to decide which video to play next on auto play,
              and Facebook uses it to recommend pages to like and people to follow.
"""

## Importing modules ##
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

## 1 | Data Preprocessing ##
"""Prepare data before training"""

# 1.1 Load dataset
credits = pd.read_csv("Dataset/tmdb_credits.csv")
movies  = pd.read_csv("Dataset/tmdb_movies.csv")
print(f"Credits Shape : {credits.shape}")
print(f"Movies Shape  : {movies.shape}")
print(f"\nCredits dataset head : \n{credits.head()}")
print(f"Movies dataset head : \n{movies.head()}\n")

# 1.2 Reanem movie_id column to id and merge it to movies dataframe
credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_merge = movies.merge(credits_column_renamed, on='id')
print(f"Merged dataset head : \n{movies_merge.head()}\n")

# 1.3 Drop unnecessary columns
movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])
print(f"Cleaned dataset head : \n{movies_cleaned.head()}\n")

## 2 | Model Creation ##
"""Create model to fit it to the data"""

# 2.1 Create vectorizer
tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',
                      analyzer='word',token_pattern=r'\w{1,}',
                      ngram_range=(1, 3),
                      stop_words = 'english')

# 2.2 Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(movies_cleaned['overview'].values.astype('U'))
print(f"tfv matrix : \n{tfv_matrix}\n")

# 2.3 Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
print(f"Sigmoid kernal : \n{sig[0]}\n")

# 2.4 Reverse mapping of indices and movie titles
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()
print(f"Indices : \n{indices}\n")

# 2.5 Give recommendations
def give_recommendations(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_cleaned['original_title'].iloc[movie_indices]

# 2.6 Test recommendation system
movie_name = "Avatar"
print(f"The recommendations for {movie_name} Movie is : \n{give_recommendations(movie_name)}")
