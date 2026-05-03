import json
import re

def load_authorized_plates():
    try:
        with open(path = "logs/authorized_plates.json") as f:
            data = json.load(f)
            return set(data.get("plates", []))
    
    except:
        return set()
    
def is_authorized(plate_text,authorized_set):
    return plate_text in authorized_set

def clean_plate_text(text):
    # keep only A-Z and 0-9
    text = re.sub(r'[^A-Z0-9]', '', text.upper())

    # enforce min length
    if len(text) < 5:
        return None

    return text