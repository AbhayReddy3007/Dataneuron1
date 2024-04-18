from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))
nli_model = SentenceTransformer('bert-base-nli-mean-tokens')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']

        # Calculate similarity score
        embeddings = nli_model.encode([text1, text2])
        similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        similarity_score = np.round(similarity_score, 4)

        return render_template('index.html', similarity_score=similarity_score)

    return render_template('index.html')

#if __name__ == '__main__':
#    app.run(debug=True)