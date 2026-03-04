import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load dataset
df = pd.read_csv("tmdb_5000_movies.csv")

df = df[['title', 'genres', 'keywords', 'overview']]
df.dropna(inplace=True)

# 2. Convert JSON columns
def convert(text):
    result = []
    for item in ast.literal_eval(text):
        result.append(item['name'])
    return " ".join(result)

df['genres'] = df['genres'].apply(convert)
df['keywords'] = df['keywords'].apply(convert)

# 3. Combine features
df['combined'] = df['genres'] + " " + df['keywords'] + " " + df['overview']

# 4. TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['combined'])

# 5. Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 6. SAVE SAFE FILES

# Save only movie titles as pure Python list
movie_titles = df['title'].tolist()
pickle.dump(movie_titles, open("movies.pkl", "wb"))

# Save similarity matrix (numpy array)
pickle.dump(cosine_sim, open("similarity.pkl", "wb"))

print("✅ Training completed successfully!")