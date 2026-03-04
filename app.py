from flask import Flask, render_template, request
import pickle

app = Flask(__name__)




movie_titles = pickle.load(open("movies.pkl", "rb"))
cosine_sim = pickle.load(open("similarity.pkl", "rb"))

def recommend(movie_title):
    movie_title = movie_title.lower()

    movie_titles_lower = [title.lower() for title in movie_titles]

    if movie_title not in movie_titles_lower:
        return ["Movie not found!"]

    index = movie_titles_lower.index(movie_title)

    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended = []
    for i in similarity_scores[1:6]:
        recommended.append(movie_titles[i[0]])

    return recommended

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    movie_name = ""

    if request.method == "POST":
        movie_name = request.form.get("movie")
        recommendations = recommend(movie_name)

    return render_template("index.html",
                           recommendations=recommendations,
                           movie_name=movie_name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)