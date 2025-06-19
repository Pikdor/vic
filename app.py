from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    items = data.get('items', [])
    n_clusters = data.get('clusters', 4)

    if len(items) < 2:
        return jsonify({'error': 'Недостаточно предметов (min 2)'}), 400

    embeddings = model.encode(items)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embeddings)

    result = {}
    for idx, label in enumerate(labels):
        key = f"cluster_{label + 1}"
        result.setdefault(key, []).append(items[idx])

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
