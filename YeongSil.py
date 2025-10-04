import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt

from google import genai
from google.genai import types

from vedo import Points, show

from config import GEMINI_KEY

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class YeongSil:
    def __init__(self):
        self.gemini = genai.Client(api_key=GEMINI_KEY)

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        self.midas = torch.hub.load("intel-isl/MiDaS", 'DPT_Hybrid')
        self.midas.to(device)
        self.midas.eval()

    # Takes some image and returns a description of the image from gemini and angle buckets of average depth
    def __process_image(self, image_path: str) -> tuple[str, list[float]]:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # img = cv2.imread(image_path)[:, :, ::-1]
        # img = cv2.resize(img, (384, 384)) 
        # img = torch.from_numpy(img).permute(2,0,1).float() / 255.0 # tensor type shit

        # # Apply the transform properly - the transform expects a PIL image or numpy array
        # img_np = img.permute(1, 2, 0).numpy()  # Convert back to HWC format for transform

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        h, w = output.shape

        x = np.flip(np.tile(np.arange(w), h))/40
        x = x - x.mean()
        y = -np.flip(output.flatten())/40 + 38
        #y = np.flip(output.flatten())
        z = np.repeat(np.arange(h), w)/40

        xyz = np.stack((x, y, z), axis=1)

        y = xyz[:, 1]

        desc = self.gemini.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data      = image_bytes,
                    mime_type = 'image/jpeg',
                ),
                'Describe what is in the image, including positions of objects, referring to it as "your view"'
            ]
        )

        # Calculate angle buckets of average depth
        # Right now, there is a 3d projection of an environment represented by the variable xyz (collection of points)located quadrants 1 and 2 of the x-y plane. y axis is depth. x and z are side to side and up and down of the original image. find the angle of each point from the x=0 plane and have "buckets" of 10 degrees each. ex. buckets shoudl be 
        # [-90 to -80, -80 to -70, -70 to -60, -60 to -50, -50 to -40, -40 to -30, -30 to -20, -20 to -10, -10 to 0, 0 to 10, 10 to 20, 20 to 30, 30 to 40, 40 to 50, 50 to 60, 60 to 70, 70 to 80, 80 to 90]
        
        x_coords = xyz[:, 0]  # x coordinates
        y_coords = xyz[:, 1]  # y coordinates (depth)
        
        angles = np.degrees(np.arctan2(x_coords, y_coords))
        
        bucket_edges = np.arange(-90, 91, 10)  # [-90, -80, -70, ..., 80, 90]
        depth_buckets = []
        
        for i in range(len(bucket_edges) - 1):
            lower_bound = bucket_edges[i]
            upper_bound = bucket_edges[i + 1]
            
            # Find points in this angle range
            mask = (angles >= lower_bound) & (angles < upper_bound)
            
            if np.any(mask):
                # Calculate average depth for points in this bucket
                avg_depth = np.mean(y_coords[mask])
            else:
                # No points in this bucket
                avg_depth = 0.0
                
            depth_buckets.append(avg_depth)

        return desc.text, depth_buckets # Type shit

    
    def get_guidance(self, image_path: str):
        return self.__process_image(image_path)

