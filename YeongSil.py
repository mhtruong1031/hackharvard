from config import GEMINI_KEY

from google import genai
from google.genai import types

class YeongSil:
    def __init__(self):
        self.gemini = genai.Client()
        pass

    # Takes some image and returns a description of the image from gemini and angle buckets of average depth
    def __process_image(self, image_path: str) -> tuple[str, list[float]]:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        desc = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data      = image_bytes,
                    mime_type = 'image/jpeg',
                ),
                'Caption this image.'
            ]
        )

        # midas model type shit

    
    def get_guidance(self, image_path: str):
        pass

