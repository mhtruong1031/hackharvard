import cv2
import torch
import numpy as np
import time
from google import genai
from google.genai import types
from config import GEMINI_KEY

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class YeongSil:
    def __init__(self):
        self.gemini = genai.Client(api_key=GEMINI_KEY)

        # Use faster MiDaS model for better performance
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform  # Faster transform
        self.midas = torch.hub.load("intel-isl/MiDaS", 'MiDaS_small')  # Faster model
        self.midas.to(device)
        self.midas.eval()

    # Takes some image and returns a description of the image from gemini and angle buckets of average depth
    def __process_image(self, image_path: str) -> tuple[str, list[float]]:
        start_time = time.time()
        print(f"[{0:.1f}s] Starting YeongSil image processing...")
        
        # Load and prepare image
        step_start = time.time()
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        print(f"[{time.time() - start_time:.1f}s] Image loaded from disk")

        img = cv2.imread(image_path)
        # Resize image for faster processing (optional: adjust size as needed)
        img = cv2.resize(img, (256, 256))  # Smaller size for faster processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(device)
        print(f"[{time.time() - start_time:.1f}s] Image preprocessed and transformed")
        
        # Depth estimation
        step_start = time.time()
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        print(f"[{time.time() - start_time:.1f}s] Depth estimation completed")

        output = prediction.cpu().numpy()
        print(f"[{time.time() - start_time:.1f}s] Depth data moved to CPU")

        h, w = output.shape

        # Vectorized coordinate generation (much faster)
        step_start = time.time()
        x_coords = np.flip(np.tile(np.arange(w), h)) / 40
        x_coords = x_coords - x_coords.mean()
        y_coords = -np.flip(output.flatten()) / 80 + 35
        z_coords = np.repeat(np.arange(h), w) / 40

        xyz = np.column_stack((x_coords, y_coords, z_coords))
        print(f"[{time.time() - start_time:.1f}s] 3D coordinates generated")

        # Commented out 3D visualization to prevent GUI crashes in background processing
        # pts = Points(xyz, r=4)  # r is point radius
        # pts.cmap("viridis", xyz[:, 1])  # color by y-values (you can change this)
        # show(pts, axes=1, bg='white', title='3D Point Cloud')

        # Gemini image description
        step_start = time.time()
        desc = self.gemini.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data      = image_bytes,
                    mime_type = 'image/jpeg',
                ),
                'Describe what is in the image, including positions of large/major objects (far left, left, middle, right, far right), referring to it as "your view" in 5 sentences.'
            ]
        )
        print(f"[{time.time() - start_time:.1f}s] Gemini image description completed")

        # Vectorized angle bucket calculation (much faster)
        step_start = time.time()
        angles = np.degrees(np.arctan2(x_coords, y_coords))
        
        # Use digitize for faster bucketing
        bucket_edges = np.arange(-90, 91, 10)
        bucket_indices = np.digitize(angles, bucket_edges) - 1
        
        # Ensure indices are within valid range
        bucket_indices = np.clip(bucket_indices, 0, len(bucket_edges) - 2)
        
        # Vectorized average calculation for each bucket
        depth_buckets = []
        for i in range(len(bucket_edges) - 1):
            mask = bucket_indices == i
            if np.any(mask):
                depth_buckets.append(np.mean(y_coords[mask]))
            else:
                depth_buckets.append(0.0)
        print(f"[{time.time() - start_time:.1f}s] Depth buckets calculated")

        print(f"[{time.time() - start_time:.1f}s] YeongSil processing completed successfully")
        return desc.text, depth_buckets

    
    def get_guidance(self, image_path: str):
        start_time = time.time()
        print(f"[{0:.1f}s] Starting YeongSil guidance generation...")
        
        desc, depth_buckets = self.__process_image(image_path)
        print(f"[{time.time() - start_time:.1f}s] Image processing completed, generating guidance...")

        depth_desc = '\n'.join([f'{dist} degrees is {depth_buckets[i]:.2f} units away.' for i, dist in enumerate(range(-85, 85, 10))])

        guidance = self.gemini.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                f'Please advise the blind user on how to traverse the following environment: {desc}.',
                f'The depth values are from 0-180 degrees, and how many units forward there are of traversable distances, where 90 degrees is directly forward, but please refer to them as 0-90 degrees right or left of forward: {depth_desc}.',
                'Please advise in 1-2 sentences with environmental context, with instructions first.'
            ]
        )
        print(f"[{time.time() - start_time:.1f}s] Navigation guidance generated successfully")

        return guidance.text, depth_buckets