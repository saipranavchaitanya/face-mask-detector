import cv2
import numpy as np
from keras.models import load_model

# Load face detector model (Caffe)
face_net = cv2.dnn.readNetFromCaffe(
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)


# Load mask detector model
mask_model = load_model("mask_detector.model")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Prepare input blob for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Loop through detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            # Get coordinates of face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Ensure coordinates are within frame size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Preprocess face for mask detection
            face_input = cv2.resize(face, (224, 224))
            face_input = face_input.astype("float") / 255.0
            face_input = np.expand_dims(face_input, axis=0)

            (mask, no_mask) = mask_model.predict(face_input)[0]

            label = "Mask" if mask > no_mask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw bounding box and label
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
