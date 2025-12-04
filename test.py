from io import BytesIO

import requests
from PIL import Image


def get_image(file_path: str) -> bytes:
    img = Image.open(file_path)
    img = img.convert("RGB")

    buffer = BytesIO()
    # Save in WEBP for smaller size; server reads bytes and opens via PIL
    img.save(buffer, format='WEBP', quality=85)
    buffer.seek(0)

    return buffer.getvalue()


url = "http://localhost:8006/predict"
image_webp = get_image('test3.jpeg')

files = {"image": ("test.webp", image_webp, "image/webp")}
response = requests.post(url, files=files, timeout=30)#.json()
print(response)
print(response.json())
