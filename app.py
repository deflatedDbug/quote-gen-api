from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import uuid
import os
import pandas as pd
import random
import string
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

app = Flask(__name__)

model_weights = "./runs/train/weights/last.pt"
current_working_directory = os.getcwd()
model_weights_path = os.path.join('runs', 'detect','train', 'weights', 'best.pt')
print("Trying to load model from:", model_weights_path)
model = YOLO(model_weights_path)

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

price_list_standard = {
    "standard-side": Decimal('231'),
    "deep-side": Decimal('231'),
    "standard-seat": Decimal('535.50'),
    "deep-seat": Decimal('535.50'),
    "angled-side": Decimal('231'),
    "angled-deep-side": Decimal('231'),
    "rollarm-side": Decimal('231'),
    "wedge-seat": Decimal('535.50'),
}
    
price_list_lovesoft = {
    "standard-side": Decimal('231'),
    "deep-side": Decimal('231'),
    "standard-seat": Decimal('675.00'),
    "deep-seat": Decimal('675.00'),
    "angled-side": Decimal('231'),
    "angled-deep-side": Decimal('231'),
    "rollarm-side": Decimal('231'),
    "wedge-seat": Decimal('675.00'),
}



def generate_quote_id():
    timestamp = datetime.now().strftime("%f")[-2:]
    
    random_part = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    
    return timestamp + random_part


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-quote', methods=['POST'])

def generate_quote_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return 'No Image selected', 400
    
    unique_filename = str(uuid.uuid4()) + ".jpg"

    dir_path = r"C:\Users\Subin Lebow\Desktop\quote-gen-api\uploaded_images"

    os.makedirs(dir_path, exist_ok=True)
    image_path = os.path.join(dir_path, unique_filename)
    image = request.files['image']
    image.save(image_path)

    results = model(image_path, save=True)
    detections = get_pandas(results)
    
    quote_id = generate_quote_id()
    discount_percent_input = request.form.get('discount_percent') or '0'
    try:
        discount_percent = Decimal(discount_percent_input)
    except InvalidOperation:
        discount_percent = Decimal('0')
    quote = generate_quote_from_detections(detections)

    client_firstName = request.form.get('client_firstName')
    client_lastName = request.form.get('client_lastName')
    clients_email = request.form.get('clients_email')
    client_streetAddress = request.form.get('client_streetAddress')
    client_city = request.form.get('client_city')
    client_state = request.form.get('client_state')
    client_zip = request.form.get('client_zip')
    
    subtotal = Decimal(quote['subtotal'])
    
    if discount_percent > Decimal('0'): 
        discount_value = (subtotal * discount_percent) / Decimal('100')
        subtotal_after_discount = subtotal - discount_value
    else:
        discount_value = Decimal('0')
        subtotal_after_discount = subtotal
    
    tax_rate = Decimal('1.07')
    tax_amount = (subtotal_after_discount * (tax_rate - Decimal('1'))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    items = quote['items']
    total = subtotal_after_discount + tax_amount
    created_date = datetime.now().strftime("%B %d, %Y")
    
    formatted_subtotal = subtotal.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    formatted_total = total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    formatted_discount = discount_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    os.remove(image_path)
    
    return render_template('quote_template.html', items=quote['items'], subtotal = formatted_subtotal, total = formatted_total, created_date=created_date, client_firstName=client_firstName, client_lastName=client_lastName, client_streetAddress=client_streetAddress, clients_email=clients_email, client_city=client_city, client_state = client_state, client_zip=client_zip, quote_id=quote_id, price_option=quote['price_option'], discount_value=formatted_discount, tax_amount=tax_amount)

def get_pandas(results):

    if isinstance(results, list) and len(results) > 0:
        results = results[0]
    boxes_data = results.boxes.data.cpu().numpy()
    columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id']
    df = pd.DataFrame(boxes_data, columns=columns)

    df['class_name'] = df['class_id'].apply(lambda x: results.names[int(x)]) 

    return df

def generate_quote_from_detections(detections):

    item_counts = {}
    price_option = request.form.get('price_option', 'standard').capitalize()
    
    if price_option == "lovesoft":
        price_list = price_list_lovesoft
    else: 
        price_list = price_list_standard

    for index, row in detections.iterrows():
        label = row['class_name']
        if label in price_list:
            if label not in item_counts:
                item_counts[label] = 0
            item_counts[label] += 1
    
    quote_items = []
    for item, count in item_counts.items():
        formatted_item = format_item_name(item)
        item_total = count * price_list[item]
        item_total_tax = item_total * Decimal('1.07')
        quote_items.append({'name': formatted_item, 'quantity': count, 'price': Decimal(item_total)})
    
    tax = item_total_tax
    subtotal = sum(item['price'] for item in quote_items)
    total = subtotal * Decimal('1.07')
    
    return {
        'items': quote_items,
        'subtotal': subtotal.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'taxes':tax.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'total': total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'price_option': price_option
    }

def format_item_name(item_name):
    parts = item_name.split('-')
    formatted_parts = [part.capitalize() for part in parts]
    return '-'.join(formatted_parts)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)