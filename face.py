import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
import os

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Load face detector and embedder
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known face embeddings
known_embeddings = []
known_names = []

known_dir = 'known_faces'  # Make sure this folder exists with images

if not os.path.exists(known_dir):
    print(f"Folder '{known_dir}' not found. Please create it and add some images.")
    exit()

for file in os.listdir(known_dir):
    if file.lower().endswith(('.jpg', '.png')):
        path = os.path.join(known_dir, file)
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)
        if face is not None:
            embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
            known_embeddings.append(embedding)
            known_names.append(os.path.splitext(file)[0])
        else:
            print(f"Face not detected in {file}, skipping.")

# Start webcam
cap = cv2.VideoCapture(0)

frame_count = 0
recognized_name = "Scanning..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize image for faster processing
    small_img = cv2.resize(img_rgb, (0, 0), fx=0.5, fy=0.5)

    if frame_count % 5 == 0:  # Process every 5th frame
        face = mtcnn(small_img)
        if face is not None:
            embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()[0]

            # Compare to known embeddings
            min_dist = float('inf')
            name = "Unknown"

            for emb, known_name in zip(known_embeddings, known_names):
                dist = np.linalg.norm(emb - embedding)
                if dist < 0.9:  # Distance threshold
                    if dist < min_dist:
                        min_dist = dist
                        name = known_name
            recognized_name = name
        else:
            recognized_name = "No Face Detected"

    # Show name
    cv2.putText(frame, recognized_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2)

    cv2.imshow('Face Recognition - Optimized', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
