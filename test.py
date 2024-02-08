import os
import hashlib

def calculate_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def remove_duplicate_images(folder_path):
    hash_dict = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = calculate_hash(file_path)
            
            if file_hash in hash_dict:
                print(f"Duplicate found: {file_path} and {hash_dict[file_hash]}")
                os.remove(file_path)  # Elimina el archivo duplicado
            else:
                hash_dict[file_hash] = file_path

# Uso de la función
remove_duplicate_images('./plates/')