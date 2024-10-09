import cv2
from ultralytics import YOLO
import numpy as np
import statistics
import torch
import random

# Carrega o modelo pré-treinado
model = torch.hub.load("ultralytics/yolov8", "yolov8l.pt")

# Inicializa a câmera do computador
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Realiza a predição
    results = model.predict(frame, classes = [0])

    # Verifica cada resultado que aparece
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]

            # Qual carta está sendo predita
            label = result.names[int(box.cls)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 8)
            cv2.putText(frame, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4)
            cv2.putText(frame, f'{label}', (x1 - 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 8)
            cv2.putText(frame, f'{label}', (x1 - 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4)

    cv2.imshow('Truco Machine', frame)
    cv2.waitKey(50)
    
cap.release()
cv2.destroyAllWindows()