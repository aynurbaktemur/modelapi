from flask import Flask, request, jsonify
import joblib
import numpy as np
import cv2
from skimage.feature import hog

app = Flask(__name__)

# MODELİ YÜKLE
MODEL_PATH = "complete_abc_model.pkl"  # Model dosya adı burada
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
scaler = model_data["scaler"]
selected_features = model_data["selected_features"]
hog_params = model_data["hog_params"]

# HOG çıkarım fonksiyonu (modeldekiyle birebir aynı olmalı!)
def extract_hog_features(img):
    return hog(img,
               orientations=hog_params["orientations"],
               pixels_per_cell=hog_params["pixels_per_cell"],
               cells_per_block=hog_params["cells_per_block"],
               transform_sqrt=hog_params["transform_sqrt"],
               block_norm=hog_params["block_norm"],
               feature_vector=True)

# Görüntüyü oku ve işleyip özellik çıkar
def process_image(file_storage):
    file_bytes = file_storage.read()
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None, "Görüntü okunamadı"

    img = cv2.resize(img, model_data["metadata"]["input_shape"])

    # HOG çıkar
    features = extract_hog_features(img)

    # Ölçekle ve seçilen özellikleri uygula
    features_scaled = scaler.transform([features])
    selected = features_scaled[:, selected_features]

    return selected, None

# Ana kontrol endpointi
@app.route('/')
def home():
    return "API çalışıyor!"

# Tahmin endpointi
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Görüntü dosyası eksik'}), 400

    file = request.files['image']
    features, error = process_image(file)

    if error:
        return jsonify({'error': error}), 400

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return jsonify({
        'tahmin': model_data['metadata']['class_names'][prediction],
        'hasar_orani': round(float(probability) * 100, 2)
    })

# Sağlık kontrolü
@app.route('/health', methods=["GET"])
def get_health():
    return {"status": "running"}

# Uygulama başlat
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
