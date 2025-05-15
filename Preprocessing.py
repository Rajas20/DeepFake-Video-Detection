import os
import glob
import cv2
import numpy as np
import face_recognition
from tqdm import tqdm

def get_video_files(directory, min_frames=150):
    video_files = glob.glob(os.path.join(directory, '*.mp4'))
    valid_videos = []
    frame_counts = []
    
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if frame_count >= min_frames:
            valid_videos.append(video_file)
            frame_counts.append(frame_count)
    
    print("Total number of videos:", len(valid_videos))
    print("Average frames per video:", np.mean(frame_counts) if frame_counts else 0)
    return valid_videos

def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success, image = vidObj.read()
    while success:
        yield image
        success, image = vidObj.read()
    vidObj.release()

def create_face_videos(video_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    existing_videos = glob.glob(os.path.join(output_dir, '*.mp4'))
    print("No of videos already present:", len(existing_videos))
    
    for path in tqdm(video_paths):
        out_path = os.path.join(output_dir, os.path.basename(path))
        if os.path.exists(out_path):
            print("File already exists:", out_path)
            continue
        
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (112, 112))
        frames = []
        
        for idx, frame in enumerate(frame_extract(path)):
            if idx > 150:
                break
            frames.append(frame)
            if len(frames) == 4:
                faces = face_recognition.batch_face_locations(frames)
                for i, face in enumerate(faces):
                    if face:
                        top, right, bottom, left = face[0]
                        try:
                            face_crop = cv2.resize(frames[i][top:bottom, left:right, :], (112, 112))
                            out.write(face_crop)
                        except Exception as e:
                            print("Error processing frame:", e)
                frames = []
        out.release()

if __name__ == "__main__":
    input_directory = "E:\Hack VDO\V1\Celeb-real"  # Change this path accordingly
    output_directory = "E:\Hack VDO\V1\Output"
    video_files = get_video_files(input_directory)
    create_face_videos(video_files, output_directory)