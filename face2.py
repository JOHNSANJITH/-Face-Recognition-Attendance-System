import cv2
import torch
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# Initialize FaceNet model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load Haar cascade for fast face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known face embeddings
known_embeddings = []
known_names = []

known_dir = 'known_faces'
for file in os.listdir(known_dir):
    if file.lower().endswith(('.jpg', '.png')):
        img = cv2.imread(os.path.join(known_dir, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.resize(face, (160, 160)))
            face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float() / 255.0
            embedding = model(face_tensor.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
            known_embeddings.append(embedding)
            known_names.append(os.path.splitext(file)[0])
            break  # Only one face per image

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (160, 160))
        face_pil = Image.fromarray(face_resized)
        face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float() / 255.0
        embedding = model(face_tensor.unsqueeze(0).to(device)).detach().cpu().numpy()[0]

        # Compare to known embeddings
        min_dist = float('inf')
        name = "Unknown"
        for known_emb, known_name in zip(known_embeddings, known_names):
            dist = np.linalg.norm(embedding - known_emb)
            if dist < 0.9 and dist < min_dist:
                min_dist = dist
                name = known_name

        # Draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
