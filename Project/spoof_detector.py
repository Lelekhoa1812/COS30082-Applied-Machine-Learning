from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import numpy as np
import cv2
import os

class SpoofDetector:
    def __init__(self, model_dir='./resources/anti_spoof_models', device_id=0):
        self.model = AntiSpoofPredict(device_id)
        self.cropper = CropImage()
        self.model_dir = model_dir

    def check_spoof(self, image):
        image_bbox = self.model.get_bbox(image)
        prediction = np.zeros((1, 3))
        for model_name in ["2.7_80x80_MiniFASNetV2.pth", 
                           "4_0_0_80x80_MiniFASNetV1SE.pth"]:
            h_input, w_input, _, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            img_cropped = self.cropper.crop(**param)
            model_path = os.path.join(self.model_dir, model_name)
            pred = self.model.predict(img_cropped, model_path)
            prediction += pred
        label = np.argmax(prediction)
        value = prediction[0][label] / 3
        is_real = (label == 1)
        return is_real, value
