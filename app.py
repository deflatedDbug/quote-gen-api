from flask import Flask, request, jsonify
from ultralytics import YOLO
import uuid
import cv2
import os
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

app = Flask(__name__)

model_weights = r"C:\Users\Subin Lebow\Desktop\yoloV8\runs\detect\train\weights\best.pt"
model = YOLO(model_weights)

class_names = {
    0: 'standard-seat',
    1: 'deep-seat',
    2: 'standard-side',
    3: 'deep-side',
    4: 'angled-side',
    5: 'angled-deep-side',
    6: 'rollarm-side',
    7: 'wedge-seat',
}

def get_pandas(results):

    if isinstance(results, list) and len(results) > 0:
        results = results[0]
    boxes_data = results.boxes.data.cpu().numpy()
    columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id']
    df = pd.DataFrame(boxes_data, columns=columns)
    
    # Map class_id to class_name (Make sure to replace 'class_names' with actual names from the model
    df['class_name'] = df['class_id'].apply(lambda x: results.names[int(x)])  # 'class_names' needs to be defined as shown before

    return df

@app.route('/generate-quote', methods=['POST'])
def generate_quote_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    
    unique_filename = str(uuid.uuid4()) + ".jpg"

    dir_path = r"C:\Users\Subin Lebow\Desktop\quote-gen-api\uploaded_images"

    os.makedirs(dir_path, exist_ok=True)
    image_path = os.path.join(dir_path, unique_filename)
    image = request.files['image']
    image.save(image_path)

    results = model(image_path, save=True)
    print(type(results))
    print(results)
    detections = get_pandas(results)

    quote = generate_quote_from_detections(detections)

    os.remove(image_path)
    return jsonify(quote)

def generate_quote_from_detections(detections):
    price_list = {
    "standard-side": Decimal('199'),
    "deep-side": Decimal('199'),
    "standard-seat": Decimal('450'),
    "deep-seat": Decimal('450'),
    "angled-side": Decimal('199'),
    "angled-deep-side": Decimal('199'),
    "rollarm-side": Decimal('199'),
    "wedge-seat": Decimal('450'),
    }

    item_counts = {}

    for index, row in detections.iterrows():
        label = row['class_name']
        if label in price_list:
            if label not in item_counts:
                item_counts[label] = 0
            item_counts[label] += 1
    
    quote = {
        'items' : [],
        'subtotal' : Decimal('0'),
        'total': Decimal('0')
    }
    for item, count in item_counts.items():
        item_total = count * price_list[item]
        item_total_with_tax = item_total * Decimal('1.07')
        quote['items'].append({'name': item, 'quantity': count, 'price': float(item_total)})
        quote['subtotal'] += item_total
        quote['total'] += item_total_with_tax

    quote['subtotal'] = float(quote['subtotal'].quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    quote['total'] = float(quote['total'].quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    return quote

if __name__ == '__main__':
    app.run(debug=True)