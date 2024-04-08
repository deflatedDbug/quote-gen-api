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

cover_mapping = {
    'standard-seat': 'Standard-Seat-Cover',
    'deep-seat': 'Deep-Seat-Cover',
    'standard-side': 'Standard-Side-Cover',
    'deep-side': 'Deep-Side-Cover',
    'angled-side': 'Angled-Side-Cover',
    'angled-deep-side': 'Angled-Deep-Side-Cover',
    'rollarm-side': 'Rollarm-Side-Cover',
    'wedge-seat': 'Wedge-Seat-Cover'
}

price_list_standard = {
    "standard-side": Decimal('225'),
    "deep-side": Decimal('225'),
    "standard-seat": Decimal('450'),
    "deep-seat": Decimal('450'),
    "angled-side": Decimal('225'),
    "angled-deep-side": Decimal('225'),
    "rollarm-side": Decimal('225'),
    "wedge-seat": Decimal('450'),
}
    
price_list_lovesoft = {
    "standard-side": Decimal('225'),
    "deep-side": Decimal('225'),
    "standard-seat": Decimal('650'),
    "deep-seat": Decimal('650'),
    "angled-side": Decimal('225'),
    "angled-deep-side": Decimal('225'),
    "rollarm-side": Decimal('225'),
    "wedge-seat": Decimal('650'),
}

fabric_velvet = {
    'Standard-Seat-Cover': Decimal('315'),
        'Deep-Seat-Cover': Decimal('315'),
        'Wedge-Seat-Cover': Decimal('315'),
        'Angled-Side-Cover': Decimal('105'),
        'Standard-Side-Cover': Decimal('105'),
        'Deep-Side-Cover': Decimal('105'),
        'Angled-Deep-Side-Cover': Decimal('105'),
        'Rollarm-Side-Cover': Decimal('105')
}

fabric_chenille = {
         'Standard-Seat-Cover': Decimal('270'),
        'Deep-Seat-Cover': Decimal('270'),
        'Wedge-Seat-Cover': Decimal('270'),
        'Angled-Side-Cover': Decimal('90'),
        'Standard-Side-Cover': Decimal('90'),
        'Deep-Side-Cover': Decimal('90'),
        'Angled-Deep-Side-Cover': Decimal('90'),
        'Rollarm-Side-Cover': Decimal('90')    
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
    
    quote = generate_quote_from_detections(detections)

    client_firstName = request.form.get('client_firstName')
    client_lastName = request.form.get('client_lastName')
    clients_email = request.form.get('clients_email')
    client_streetAddress = request.form.get('client_streetAddress')
    client_city = request.form.get('client_city')
    client_state = request.form.get('client_state')
    client_zip = request.form.get('client_zip')
        
    formatted_subtotal = quote['subtotal'].quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    formatted_total = quote['total'].quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    formatted_discount = quote['discount'].quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    formatted_tax = quote['taxes'].quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    comma_separated_discount = "{:,.2f}".format(formatted_discount)
    comma_separated_subtotal = "{:,.2f}".format(formatted_subtotal)
    comma_separated_total = "{:,.2f}".format(formatted_total)
    comma_separated_tax = "{:,.2f}".format(formatted_tax)
    
    fabric_type =quote['fabric_type']
    discount_rate = quote['discount_percent']
    created_date = datetime.now().strftime("%B %d, %Y")
    
    if fabric_type == "Velvet":
        display_fabric_name = "Corded Velvet"
    else:
        display_fabric_name = fabric_type

    os.remove(image_path)
    
    return render_template('quote_template.html', items=quote['items'], comma_separated_discount=comma_separated_discount, comma_separated_subtotal=comma_separated_subtotal, comma_separated_tax=comma_separated_tax, comma_separated_total=comma_separated_total, subtotal=formatted_subtotal , fabric_name=display_fabric_name, total = formatted_total, created_date=created_date, client_firstName=client_firstName, client_lastName=client_lastName, client_streetAddress=client_streetAddress, clients_email=clients_email, client_city=client_city, client_state = client_state, client_zip=client_zip, quote_id=quote_id, price_option=quote['price_option'], discount_rate=discount_rate, discount_value=formatted_discount, tax_amount=formatted_tax)

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
    cover_counts = {}
    
    fabric_type = request.form.get('fabric_type', 'Velvet')
    price_option = request.form.get('price_option', 'standard').capitalize()
    
    if price_option == "Lovesoft":
        price_list = price_list_lovesoft
    else: 
        price_list = price_list_standard
    
    fabric_pricing_dict = fabric_velvet if fabric_type == 'Velvet' else fabric_chenille
    
    for _, row in detections.iterrows():
        label = row['class_name']
        
        if label in price_list:
            item_counts[label] = item_counts.get(label, 0) + 1
            cover_label = cover_mapping[label]
            cover_counts[cover_label] = cover_counts.get(cover_label, 0) + 1
    
    quote_items = []
    
    for item, count in item_counts.items():
        formatted_item = format_item_name(item)
        item_total = count * price_list[item] 
        quote_items.append({'name': formatted_item, 'quantity': count, 'price': Decimal(item_total)})
    
    for cover, count in cover_counts.items():
        formatted_cover = format_item_name(cover)
        cover_total = count * fabric_pricing_dict[cover]
        quote_items.append({'name': formatted_cover, 'quantity': count, 'price': Decimal(cover_total)})
    
    subtotal = sum(item['price'] for item in quote_items)
    discount_percent_input = request.form.get('discount_percent') or '0'
    try: 
        discount_percent = Decimal(discount_percent_input)
    except InvalidOperation:
        discount_percent = Decimal ('0')
    discount_value = (subtotal * discount_percent) / Decimal('100') if discount_percent > Decimal('0') else Decimal('0')
    subtotal_after_discount = subtotal - discount_value
    tax_rate = Decimal('1.07')
    tax_amount = (subtotal_after_discount * (tax_rate - Decimal('1'))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    total = subtotal_after_discount + tax_amount
     
    return {
        'items': quote_items,
        'subtotal': subtotal_after_discount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'discount_percent': discount_percent,
        'discount': discount_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'taxes':tax_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'total': total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'price_option': price_option,
        'fabric_type': fabric_type
    }

def format_item_name(item_name):
    parts = item_name.split('-')
    formatted_parts = [part.capitalize() for part in parts]
    
    if formatted_parts[-1] == "Seat":
        formatted_parts.append("Insert")
    return ' '.join(formatted_parts)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)