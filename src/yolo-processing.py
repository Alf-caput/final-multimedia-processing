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
    data_path = "data"
    video_name = "cars-highway.mp4"
    video_path = os.path.join(data_path, video_name)

    video = Video(video_path)
    print(f"{video.name = }")
    print(f"{video.shape = }")
    print(f"{video.fps = }")

    models_path = "pretrained_models"
    model_name = "yolo11n.pt"
    yolo_path = os.path.join(models_path, model_name)

    yolo = YOLO(yolo_path, verbose=False)

    x1_roi, x2_roi = 90, 280
    y1_roi, y2_roi = 170, 290
    roi_mask = slice(y1_roi, y2_roi), slice(x1_roi, x2_roi)

    obj_positions = {}
    obj_velocities = {}
    detection_lifetime_frames = 5
   
    for i, frame in enumerate(video):
        results = yolo.track(frame[roi_mask], persist=True, classes=[2, 7], conf=0.6, iou=0.5)

        for obj in results[0].boxes:
            try:
                id = int(obj.id.item())
                x, *_, y = map(int, obj.xyxy[0].numpy()) # (x, y) is bottom-left corner of the object
                obj_pos_frame = (x, y, i)

                if id not in obj_positions.keys():
                    obj_positions[id] = [obj_pos_frame]
                    obj_velocities[id] = None
                else:
                    obj_positions[id].append(obj_pos_frame)
                    xpx_diff = obj_positions[id][-1][0] - obj_positions[id][-2][0]
                    ypx_diff = obj_positions[id][-1][1] - obj_positions[id][-2][1]
                    frame_diff = obj_positions[id][-1][-1] - obj_positions[id][-2][-1]

                    vx = xpx_diff/frame_diff
                    vy = ypx_diff/frame_diff

                    if id not in obj_velocities.keys() or obj_velocities[id] is None:
                        obj_velocities[id] = ypx_diff/frame_diff
                    else:
                        obj_velocities[id] = (obj_velocities[id] + vy) / 2 

                    speed = obj_velocities[id] * video.fps

                    cv2.putText(frame[roi_mask], f"{-speed:.2f} px/s", (x, y+7), 0, 0.5, (0, 255, 0), 1)
            

                cv2.circle(frame[roi_mask], (x, y), 3, (0, 0, 255), -1)

            except AttributeError:
                print("No objects detected, resuming...")
                continue

        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 0, 255), 2)

        vehicle_count = len(obj_positions)
        cv2.putText(frame, f"Vehicle count: {vehicle_count}", (x2_roi+5, y1_roi-5), 0, 0.5, (0, 0, 255), 1)

        if obj_velocities:
            filtered = [value for value in obj_velocities.values() if value is not None] # Remove None's
            avg_speed = np.mean(filtered)
            cv2.putText(frame, f"Avg speed: {-avg_speed*video.fps:.2f}px/s", (x1_roi-5, y1_roi-5), 0, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f"Traffic: {-avg_speed*video.fps<15 and vehicle_count>3}", (20, 20), 0, 0.5, (255, 0, 0), 1)

        cv2.imshow("YOLO", frame)

        for id in list(obj_positions.keys()):
            if i - obj_positions[id][-1][-1] > detection_lifetime_frames:
                del obj_positions[id]
                del obj_velocities[id]
    
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
