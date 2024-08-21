import requests
from pathlib import Path

API_KEY = '42241795-b2b497c4d0091f2b319ac9ee1'
QUERY = 'fondos de colores abstractos'
URL = f"https://pixabay.com/api/"

params = {
  'key': API_KEY,
  'q': QUERY,
  'image_type': 'photo',
  'pretty': 'true'
}

response = requests.get(URL, params=params)
data = response.json()

if int(data['totalHits']) > 0:
    for hit in data['hits']:
        print(hit['pageURL'])
else:
    print('No hits')


def download_images(url_links:list):
    script_dir = Path(__file__).parent.absolute()
    folder_images = script_dir / "images"
    index = max([0].extend(int(img.name.split('img')[0].split('.')[0]) for img in folder_images.iterdir()))

    for image in url_links:
        index += 1
        r = requests.get(image, allow_redirects=False)
        file_name = f"istanbul_image_{index}.jpg"
        fiel_path = folder_images / file_name
        open(fiel_path, 'wb').write(r.content)
