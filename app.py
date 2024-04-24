from flask import Flask, request, jsonify, render_template, redirect, url_for
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

detections_store = {}
fabric_type_global = None
price_option_global = None
discount_global = Decimal('0')
quote_id_global = None
client_firstName_global = None
client_lastName_global = None
clients_email_global = None
client_streetAddress_global = None
client_city_global = None
client_state_global = None
client_zip_global = None
discount_rate_global = Decimal('0')
date_global = None

def get_pandas(results):

    if isinstance(results, list) and len(results) > 0:
        results = results[0]
    boxes_data = results.boxes.data.cpu().numpy()
    columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id']
    df = pd.DataFrame(boxes_data, columns=columns)

    df['class_name'] = df['class_id'].apply(lambda x: results.names[int(x)]) 

    return df

def generate_quote_from_detections(detections, quote_id):
    item_counts = {}
    cover_counts = {}
    global client_firstName_global
    global client_lastName_global
    global clients_email_global
    global client_streetAddress_global
    global client_city_global
    global client_state_global
    global client_zip_global
    global discount_global
    global fabric_type_global
    
    if isinstance(detections, list):
        detections = pd.DataFrame(detections)
        print("Converted DataFrame:", detections)
    
    fabric_type = request.form.get('fabric_type', 'Velvet')
    price_option = request.form.get('price_option', 'standard').capitalize()
    
    fabric_type_global = fabric_type
    price_option_global = price_option
    quote_id_global = quote_id
    
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
    
    subtotal = sum(Decimal(item['price']) for item in quote_items)
    discount_percent_input = request.form.get('discount_percent') or '0'
    discount_global = discount_percent_input
    try: 
        discount_percent = Decimal(discount_percent_input)
    except InvalidOperation:
        discount_percent = Decimal ('0')
        
    client_firstName_global = request.form['client_firstName']
    client_lastName_global = request.form['client_lastName']
    clients_email_global = request.form['clients_email']
    client_streetAddress_global = request.form.get('client_streetAddress', '')
    client_city_global = request.form.get('client_city', '')
    client_state_global = request.form.get('client_state', '')
    client_zip_global = request.form.get('client_zip', '')
        
    discount_value = (subtotal * discount_percent) / Decimal('100') if discount_percent > Decimal('0') else Decimal('0')
    tax_rate = Decimal('1.07')
    tax_amount = (subtotal * (tax_rate - Decimal('1'))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    total = subtotal + tax_amount - discount_value
    
    return {
        'items': quote_items,
        'subtotal': subtotal.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'discount_percent': discount_percent,
        'discount': discount_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'taxes':tax_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'total': total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'price_option': price_option,
        'fabric_type': fabric_type,
        'quote_id': quote_id,
        'fabric_type_global': fabric_type_global,
        'price_option_global': price_option_global,
        'quote_id_global': quote_id_global,
        'discount_global': discount_global,
        'client_firstName_global': client_firstName_global,
        'client_lastName_global': client_lastName_global,
        'clients_email_global': clients_email_global,
        'client_streetAddress_global': client_streetAddress_global,
        'client_city_global': client_city_global,
        'client_state_global': client_state_global,
        'client_zip_global': client_zip_global
    }
    
def generate_quote_id():
    timestamp = datetime.now().strftime("%f")[-2:]
    
    random_part = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    
    return timestamp + random_part

def format_item_name(item_name):
    parts = item_name.split('-')
    formatted_parts = [part.capitalize() for part in parts]
    
    if formatted_parts[-1] == "Seat":
        formatted_parts.append("Insert")
    return ' '.join(formatted_parts)

def add_to_detections_store(quote_id, item):
    if quote_id not in detections_store:
        detections_store[quote_id] = []
    detections_store[quote_id].append(item)
    
def standardize_decimal(value, precision='0.01'):
    return Decimal(value).quantize(Decimal(precision), rounding=ROUND_HALF_UP)

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
    raw_images_dir = r"C:\Users\Subin Lebow\Desktop\Lovesac-Stuff\quote-gen-api\raw_images" 
    raw_images_path = os.path.join(raw_images_dir, unique_filename) 
    image_file.save(raw_images_path)

    results = model(raw_images_path, save=False)
    detections = get_pandas(results)
    global detections_store
    quote_id = generate_quote_id()
    quote = generate_quote_from_detections(detections, quote_id)
    detections_store[quote_id] = quote
    return redirect(url_for('display_quote', quote_id=quote_id))

@app.route('/quote/<quote_id>')
def display_quote(quote_id):
    global quote_id_global
    quote_id_global = quote_id
    date = datetime.now()
    formatted_date = date.strftime('%m/%d/%Y')
    quote = detections_store.get(quote_id_global)
    
    print("display_quote_route:", quote)
    if not quote:
        return jsonify({'error': 'Quote not found'}), 404
    
    return render_template('quote_template.html', **quote, formatted_date=formatted_date)
    
@app.route('/update-quote/<quote_id_global>', methods=['POST'])

def generate_quote_from_update(quote_id_global):
    updated_quantities = {}
    for key in request.form:
        if key.startswith('quantity_'):
            item_name = key.split('quantity_')[1].replace('_', ' ')
            updated_quantities[item_name] = int(request.form[key])
            
    if quote_id_global not in detections_store:
        raise ValueError("Quote ID not found")
    
    items = detections_store[quote_id_global]['items']
    for item in items:
        if item['name'] in updated_quantities:
            new_quantity = updated_quantities[item['name']]
            unit_price = Decimal(item['price']) / item['quantity']
            item['quantity'] = new_quantity
            item['price'] = standardize_decimal(unit_price * new_quantity)
    
    print("Content of items:", items)
    subtotal = sum(Decimal(item['price']) for item in items)
    discount_percent_input = discount_global
    print("discount_percent_input percentage:", discount_percent_input)
    try: 
        discount_percent_input = Decimal(discount_percent_input)
    except InvalidOperation:
        discount_percent_input = Decimal ('0')
        
    discount_value = (subtotal * discount_percent_input) / Decimal('100') if discount_percent_input > Decimal('0') else Decimal('0')
    print("Update quote discount_value_global:", discount_value)
    print("update quote discount value from return value:", discount_global)
    
    date = datetime.now()
    formatted_date = date.strftime('%m/%d/%Y')
    
    tax_rate = Decimal('1.07')
    tax_amount = (subtotal * (tax_rate - Decimal('1'))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    total = subtotal + tax_amount - discount_value
    detections_store[quote_id_global] = {
        'items': items,
        'subtotal': subtotal.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'total': total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'discount_percent': discount_percent_input,
        'discount': discount_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'taxes': tax_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        'quote_id': quote_id_global
        
    }
    print("Return type for update quote:", detections_store[quote_id_global])
    updated_quote = detections_store[quote_id_global]
    return render_template('quote_template.html', 
                           items=updated_quote['items'],
                           quote_id=quote_id_global,
                           subtotal=updated_quote['subtotal'],
                           total=updated_quote['total'],
                           discount=updated_quote['discount'],
                           taxes=updated_quote['taxes'],
                           price_option=updated_quote.get('price_option', 'Standard'),
                           fabric_type_global= fabric_type_global,
                           discount_percent=updated_quote.get('discount_percent', 0),
                           client_firstName_global=client_firstName_global,
                           client_lastName_global=client_lastName_global,
                           clients_email_global=clients_email_global,
                           client_streetAddress_global=client_streetAddress_global,
                           client_city_global=client_city_global,
                           client_state_global=client_state_global,
                           client_zip_global=client_zip_global,
                           formatted_date=formatted_date )    

@app.route('/delete-item', methods=['POST'])
def delete_item():
    global detections_store
    global quote_id_global
    data = request.json
    itemName = data['itemName']
    quote_id = quote_id_global
    
    normalized_item_name = itemName.replace('_', ' ').title()
    
    print("Received item to delete:", normalized_item_name)
    print("Items before deletion:", detections_store.get(quote_id, []))
    
    if quote_id in detections_store:
        detections_store[quote_id] = [item for item in detections_store[quote_id] if item['name'] != normalized_item_name]
    print("Items after deletion:", detections_store.get(quote_id, []))
    return redirect(url_for('view_quote', quote_id=quote_id))

@app.route('/view-quote')
def view_quote(quote_id):
    if quote_id not in detections_store:
        return jsonify({'error': 'Quote not found'}), 404
    quote = detections_store[quote_id]
    return render_template('quote_template.html', items=quote)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)