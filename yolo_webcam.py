import torch
import cv2

label_id_to_indonesia = {
    'person': 'Orang',
    'bicycle': 'Sepeda',
    'car': 'Mobil',
    'motorcycle': 'Sepeda Motor',
    'bus': 'Bus',
    'train': 'Kereta',
    'truck': 'Truk',
    'traffic light': 'Lampu Lalu Lintas',
    'fire hydrant': 'Hidran',
    'stop sign': 'Rambu Berhenti',
    'parking meter': 'Parkir Meter',
    'bench': 'Bangku',
    'bird': 'Burung',
    'cat': 'Kucing',
    'dog': 'Anjing',
    'sheep': 'Domba',
}

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for det in results.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, cls_id = det
        cls_id = int(cls_id)
        label_eng = model.names[cls_id]
        label_ind = label_id_to_indonesia.get(label_eng, label_eng)
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, f'{label_ind} {conf:.2f}', (int(xmin), int(ymin)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('YOLO Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()