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
            self.capture.get,
            [
                cv2.CAP_PROP_FPS,
                cv2.CAP_PROP_FRAME_COUNT,
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT
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
    print(f"{video.fps = :.0f}")

    models_path = "../pretrained_models"
    model_name = "yolo11n.pt"
    yolo_path = os.path.join(models_path, model_name)

    yolo = YOLO(yolo_path)
    for frame in video:
        # frame = np.ascontiguousarray(frame[100:, :320]) # Crop the frame
        results = yolo.track(frame, persist=True, classes=[2, 7])
        
        for obj in results[0].boxes:
            cls = obj.cls.item() # .item() extracts value of tensor of a single element
            id = obj.id.item()
            x, y, *_ = map(int, obj.xywh[0].numpy()) # NOTE: (x, y) is the center of the bounding box
            # print(f"Label(cls={cls:.0f}): {yolo.names[cls]}")
            # print(f"ID: {id} -> Location: {x=:.2f}, {y=:.2f}, {w=:.2f}, {h=:.2f}")
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(frame, f"{id:.0f}", (x, y-7), 0, 0.5, (0, 0, 255), 1)

        cv2.imshow("YOLO", frame)
    
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
