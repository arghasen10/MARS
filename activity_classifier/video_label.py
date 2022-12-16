import cv2
import glob
from datetime import datetime, timedelta
video_files = glob.glob("ground_truth/2022-11-28 21-08-26.mp4")
cap = cv2.VideoCapture(video_files[0])
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
frame_count = 0
start_time = video_files[0].split(".")[0].split("/")[-1]
start_time = datetime.strptime(start_time, "%Y-%m-%d %H-%M-%S")
result = cv2.VideoWriter('filename'+str(start_time)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (int(cap.get(3)),
                                                                                                  int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1
    if ret is None:
        break
    text = start_time + timedelta(seconds=frame_count/fps)
    print(text)
    cv2.putText(frame, str(text), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA, False)
    # cv2.imshow("frame", frame)
    result.write(frame)
    if cv2.waitKey(int(fps)) & 0xFF == ord('s'):
        break
    if cv2.waitKey(int(fps)) & 0xFF == ord('p'):
        while 1:
            if cv2.waitKey(int(fps)) & 0xFF == ord('p'):
                break
            continue

cap.release()
result.release()
cv2.destroyAllWindows()
