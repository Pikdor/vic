from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Создаём zero-shot классификатор
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Пример универсальных категорий (можно менять/дополнять)
CANDIDATE_LABELS = [
    "еда", "одежда", "техника", "гигиена", "мебель", "инструменты", "бытовые товары",
    "электроника", "канцелярия", "спорт", "игрушки", "косметика", "автомобильные товары"
]

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    items = data.get("items")
    labels = data.get("labels", CANDIDATE_LABELS)
    if not items or len(items) == 0:
        return jsonify({"error": "No items provided"}), 400

    # Классифицируем все предметы сразу (батчинг)
    out = classifier(items, candidate_labels=labels)

    # Формируем результат
    result = {label: [] for label in labels}
    for item_result in out:
        top_label = item_result["labels"][0]
        item = item_result["sequence"]
        result[top_label].append(item)

    # Убираем пустые категории
    result = {k: v for k, v in result.items() if v}

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
