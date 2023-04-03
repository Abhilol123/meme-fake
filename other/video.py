import os
import cv2
from PIL import Image
import logging
import shutil

def convert_video_to_data(video_path, no_of_images, name) -> bool:
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logging.error("Error opening video file")
        return False

    no_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame in range(0, no_of_frames, int(no_of_frames / no_of_images)):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, image = video.read()
        if not success:
            return False
        pil_image = Image.fromarray(image[:,:,::-1]).convert("RGB")
        
        pil_image.save(f"{name}/{str(frame)}.png")

    shutil.make_archive(name, 'zip', name)
    logging.info(f"data saved in the folder: ./data/{name}.zip")
    return True

convert_video_to_data("Abhinav.mp4", 100, "abhinav")
