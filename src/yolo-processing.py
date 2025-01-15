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
    avg_velocities = {}
    detection_lifetime_frames = 5

    for i, frame in enumerate(video):
        results = yolo.track(frame[roi_mask], persist=True, classes=[2, 7], conf=0.45, iou=0.5, verbose=False)

        for obj in results[0].boxes:
            try:
                id = int(obj.id.item())
                x, y, *_ = map(int, obj.xywh[0].numpy())
                obj_pos_frame = (x, y, i)

                if id not in obj_positions.keys():
                    obj_positions[id] = [obj_pos_frame]
                    obj_velocities[id] = [None]
                    avg_velocities[id] = None
                else:
                    obj_positions[id].append(obj_pos_frame)
                    xpx_diff = obj_positions[id][-1][0] - obj_positions[id][-2][0] # Omitted for simplicity
                    ypx_diff = obj_positions[id][-1][1] - obj_positions[id][-2][1]
                    frame_diff = obj_positions[id][-1][2] - obj_positions[id][-2][2]

                    vx = xpx_diff/frame_diff * video.fps # Omitted for simplicity
                    vy = -ypx_diff/frame_diff * video.fps

                    if obj_velocities[id] == [None]:
                        obj_velocities[id] = [vy]
                        avg_velocities[id] = vy
                    else:
                        obj_velocities[id].append(vy)
                        avg_velocities[id] = np.mean(obj_velocities[id])
                        cv2.putText(frame[roi_mask], f"{avg_velocities[id]:.2f} px/s", (x, y+7), 0, 0.5, (0, 255, 0), 1)
            

                cv2.circle(frame[roi_mask], (x, y), 3, (0, 0, 255), -1)

            except AttributeError:
                print("Invalid object, resuming...")
                continue

        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 0, 255), 2)

        vehicle_count = len(results[0].boxes)
        cv2.putText(frame, f"Vehicle count: {vehicle_count}", (x2_roi+5, y1_roi-5), 0, 0.5, (0, 0, 255), 1)

        filtered = [value for value in avg_velocities.values() if value is not None] # Remove None's
        if filtered:
            avg_speed = np.mean(filtered)
            bool_traffic = avg_speed < 15 and vehicle_count > 3
            cv2.putText(frame, f"Traffic: {bool_traffic}", (20, 20), 0, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, f"Avg speed: {avg_speed:.2f}px/s", (x1_roi-5, y1_roi-5), 0, 0.5, (0, 255, 0), 1)

        cv2.imshow("YOLO", frame)

        for id in list(obj_positions.keys()):
            if i - obj_positions[id][-1][-1] > detection_lifetime_frames:
                del obj_positions[id]
                del obj_velocities[id]
                del avg_velocities[id]

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
