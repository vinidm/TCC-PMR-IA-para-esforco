from datetime import timedelta
from PIL import Image
import cv2
import numpy as np
import os
import shutil
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
SAVING_FRAMES_PER_SECOND = 1 # numero de frames a serem salvos por segundo

def crop_rate(img,x,y,largura,altura, largura_lfw = 96, altura_lfw = 96, interpolation=cv2.INTER_CUBIC): 
    razao_aspecto = altura_lfw/largura_lfw
    centro_x = x + largura/2
    centro_y = y + altura/2
    area = largura*altura
    largura_adj = np.sqrt(area/razao_aspecto)
    altura_adj = razao_aspecto*largura_adj
    x_min = int(np.floor(centro_x-largura_adj/2))
    x_max = int(np.ceil(centro_x+largura_adj/2))
    y_min = int(np.floor(centro_y-altura_adj/2 + 0.5))
    y_max = int(np.ceil(centro_y+altura_adj/2 + 0.5)) 
      
    if y_min < 0:
       y_max -= y_min
       y_min = 0
    if x_min < 0:
       x_max -= x_min
       x_min = 0  
      
    # Centralize and crop
    crop_img = img[y_min:y_max, x_min:x_max]
    img_lfw = cv2.resize(crop_img, (largura_lfw, altura_lfw), interpolation=interpolation)
    return img_lfw

def find_face(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return False
        image_rows, image_cols, _ = image.shape
        max_height = 0
        for detection in results.detections:
            try:
                location_data = detection.location_data
                relative_bounding_box = location_data.relative_bounding_box
                rect_start_point = _normalized_to_pixel_coordinates(relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,image_rows)
                rect_end_point = _normalized_to_pixel_coordinates(relative_bounding_box.xmin+relative_bounding_box.width, relative_bounding_box.ymin+relative_bounding_box.height, image_cols, image_rows)
                xleft,ytop = rect_start_point
                xright,ybot = rect_end_point
                width = xright-xleft
                height = ybot-ytop
                if height > max_height:
                  max_height = height
                  bounding_box = [
                      xleft, ytop,
                      width, height,
                  ] 
                return bounding_box
            except:
                return False

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def make_frames(video_file,prefix):
    filename, _ = os.path.splitext(video_file)
    # make a folder by the name of the video file
    # read the video file    
    cap = cv2.VideoCapture(video_file)
    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # start the loop
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = count / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration, 
            # then save the frame
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            frame_duration_formatted = frame_duration_formatted.replace(":","-")
            pathname = f"{prefix}_frame{frame_duration_formatted}.jpg"
            frame = cv2.rotate(frame,cv2.ROTATE_180)
            cv2.imwrite(pathname, frame)

            img = cv2.imread(pathname)
            face = find_face(img)
            if face:
                [x, y,largura,altura] = face
                gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                #print(img.size)
                person = crop_rate(gray,x=x,y=y,largura=largura,altura=altura)
                #print(img2.size)
                #break
                cv2.imwrite(pathname, person)
                # drop the duration spot from the list, since this duration spot is already saved
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            else:
                os.remove(pathname)
        # increment the frame count
        count += 1

video = os.listdir('./Videos_Frames')[0]
prefix = os.path.splitext(video)[0]
idx = os.listdir().index('Videos_Frames')
father_path = os.listdir()[idx]
path = os.path.join(father_path,video)
make_frames(path,prefix)
print('done')