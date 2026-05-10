import json

NAME_MAP = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn___Cercospora_leaf_spot",
    "Corn_(maize)___Common_rust_": "Corn___Common_rust",
    "Corn_(maize)___healthy": "Corn___healthy",
    "Grape___Esca_(Black_Measles)": "Grape___Esca",
}

classes_path = r"C:\Users\lenvo\Desktop\Krishi-ai full\models\disease_classes.json"

with open(classes_path) as f:
    classes = json.load(f)

fixed = [NAME_MAP.get(c, c) for c in classes]

with open(classes_path, 'w') as f:
    json.dump(fixed, f, indent=2)

print("Done! disease_classes.json is fixed.")