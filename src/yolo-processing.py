import os

import cv2
import numpy as np
from ultralytics import YOLO

class Video:
    def __init__(self, video_path):
        self.path = video_path
        self.name, self.extension = os.path.splitext(os.path.basename(self.path))
        self.capture = cv2.VideoCapture(video_path)
        self.fps, *self.shape = map(
            lambda prop: int(self.capture.get(prop)),
            [
                cv2.CAP_PROP_FPS,
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FRAME_COUNT,
            ]
        )
        self.capture.release()
    
    def frame_gen(self):
        self.capture = cv2.VideoCapture(self.path)

        while self.capture.isOpened() and cv2.waitKey(1) == -1:
            read_successfully, main_frame = self.capture.read()

            if read_successfully:
                yield main_frame

        self.capture.release()
    
    def __iter__(self):
        return self.frame_gen()

def main():
    data_path = "../data"
    video_name = "cars-highway.mp4"
    video_path = os.path.join(data_path, video_name)

    video = Video(video_path)
    print(f"{video.name = }")
    print(f"{video.shape = }")
    print(f"{video.fps = }")

    models_path = "../pretrained_models"
    model_name = "yolo11n.pt"
    yolo_path = os.path.join(models_path, model_name)

    yolo = YOLO(yolo_path, verbose=False)

    x1_roi, x2_roi = 100, video.shape[0]-100
    # y1_roi, y2_roi = 0, video.shape[1]
    # x1_roi, x2_roi = 100, 300
    y1_roi, y2_roi = 170, 290
    roi_mask = slice(y1_roi, y2_roi), slice(x1_roi, x2_roi)
    obj_positions = {}
   
    for i, frame in enumerate(video):
        # frame = np.ascontiguousarray(frame[100:, :320]) # Crop the frame
        results = yolo.track(frame[roi_mask], persist=True, classes=[2, 7], conf=0.6, iou=0.5)

        for obj in results[0].boxes:
            try:
                # cls = obj.cls.item() # .item() extracts value of tensor of a single element
                id = int(obj.id.item())
                x, y, *_ = map(int, obj.xywh[0].numpy()) # NOTE: (x, y) is the center of the bounding box
                obj_pos_frame = (y, i)
                if id not in obj_positions.keys():
                    obj_positions[id] = [obj_pos_frame]
                else:
                    obj_positions[id].append(obj_pos_frame)
                    px_diff = obj_positions[id][1][0] - obj_positions[id][0][0]
                    frame_diff = obj_positions[id][1][1] - obj_positions[id][0][1]
                    speed = px_diff / frame_diff
                    del obj_positions[id][0] 
                    cv2.putText(frame[roi_mask], f"{speed} pixels/frame", (x, y+7), 0, 0.5, (0, 0, 255), 1)

                cv2.circle(frame[roi_mask], (x, y), 3, (0, 0, 255), -1)
                cv2.putText(frame[roi_mask], f"{id}", (x, y-7), 0, 0.5, (0, 0, 255), 1)
            # except AttributeError:
            #     print("No objects detected, resuming...")
            #     continue
            finally:
                cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 0, 255), 2)
                cv2.imshow("YOLO", frame)
    
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
