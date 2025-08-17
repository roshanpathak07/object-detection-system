import cv2
import numpy as np
import time

# Load YOLOv4
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().ravel()]

# Assign random colors to classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Start webcam or use a video path
cap = cv2.VideoCapture(0)  # Use "video.mp4" or IP camera URL if needed

# FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing (optional)
    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]

    # Create a 4D blob from frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass
    outputs = net.forward(output_layers)

    # Initialization
    boxes = []
    confidences = []
    class_ids = []

    # Process outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # first 5 are box info
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes
    if len(indexes) > 0:
        for i in np.array(indexes).flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("YOLOv4 Detection", frame)

    # Press 'Esc' to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()