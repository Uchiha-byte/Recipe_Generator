import os
import urllib.request
import sys

def download_file(url, filename):
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        sys.exit(1)

def main():
    # Create data directory if it doesn't exist
    data_dir = os.path.join('Foodimg2Ing', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # URLs from the README
    model_url = "https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt"
    ingr_vocab_url = "https://dl.fbaipublicfiles.com/inversecooking/ingr_vocab.pkl"

    # Download files
    model_path = os.path.join(data_dir, 'modelbest.ckpt')
    ingr_vocab_path = os.path.join(data_dir, 'ingr_vocab.pkl')

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:  # Check if file exists and is not too small
        download_file(model_url, model_path)
    else:
        print("Model file already exists")

    if not os.path.exists(ingr_vocab_path) or os.path.getsize(ingr_vocab_path) < 1000:  # Check if file exists and is not too small
        download_file(ingr_vocab_url, ingr_vocab_path)
    else:
        print("Vocabulary file already exists")

if __name__ == "__main__":
    main() 