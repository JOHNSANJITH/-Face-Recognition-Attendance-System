import cv2
import torch
import numpy as np
import os
import pandas as pd
from datetime import datetime
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Setup device and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global vars
known_embeddings = []
known_names = []
marked_today = set()
known_dir = 'known_faces'
attendance_file = 'attendance.csv'

# Load known faces
def load_known_faces():
    known_embeddings.clear()
    known_names.clear()
    for file in os.listdir(known_dir):
        if file.lower().endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(known_dir, file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_tensor = torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                embedding = model(face_tensor).detach().cpu().numpy()[0]
                known_embeddings.append(embedding)
                known_names.append(os.path.splitext(file)[0])
                break

# Attendance CSV
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=['Name', 'Time', 'Date']).to_csv(attendance_file, index=False)

def mark_attendance(name):
    if name in marked_today:
        return None

    df = pd.read_csv(attendance_file)
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if not ((df['Name'] == name) & (df['Date'] == today)).any():
        new_row = pd.DataFrame([[name, time_str, today]], columns=['Name', 'Time', 'Date'])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        marked_today.add(name)
        return f"✅ Marked: {name} at {time_str}"
    else:
        marked_today.add(name)
        return f"✔️ Already marked: {name}"

# Upload to Google Sheets
def upload_to_google_sheets():
    try:
        log_box.insert(tk.END, "🌐 Starting upload process...\n")
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        client = gspread.authorize(creds)

        log_box.insert(tk.END, "🔑 Authenticated with Google Sheets API.\n")
        sheet = client.open('AttendanceSheet').sheet1

        df = pd.read_csv(attendance_file)
        log_box.insert(tk.END, f"📄 Uploading {len(df)} records...\n")

        sheet.clear()
        sheet.append_row(df.columns.tolist())
        for row in df.values.tolist():
            sheet.append_row(row)

        log_box.insert(tk.END, "✅ Upload completed.\n")
        log_box.yview(tk.END)
        messagebox.showinfo("Upload", "✅ Attendance uploaded to Google Sheets.")
    except Exception as e:
        log_box.insert(tk.END, f"❌ Upload Error: {e}\n")
        log_box.yview(tk.END)
        messagebox.showerror("Upload Error", f"❌ Failed to upload:\n{e}")

# Register new face
def register_new_face():
    name = simpledialog.askstring("Register", "Enter name:")
    if not name:
        return

    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Register", "Press 's' to capture and save the face.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Register Face - Press S to Save", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            face_file = os.path.join(known_dir, f"{name}.jpg")
            cv2.imwrite(face_file, frame)
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    load_known_faces()
    messagebox.showinfo("Registered", f"Face for '{name}' added successfully!")

# Reset session
def reset_for_new_day():
    marked_today.clear()
    log_box.insert(tk.END, "🔁 Session reset for the day.\n")
    log_box.yview(tk.END)

# GUI setup
root = tk.Tk()
root.title("Face Attendance System")
root.geometry("800x700")

video_label = tk.Label(root)
video_label.pack()

log_box = scrolledtext.ScrolledText(root, width=100, height=10, font=('Courier', 10))
log_box.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

tk.Button(btn_frame, text="🔁 Reset", command=reset_for_new_day).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="☁️ Upload to Cloud", command=upload_to_google_sheets).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="➕ Register Face", command=register_new_face).pack(side=tk.LEFT, padx=5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

load_known_faces()

# Track names already printed
printed_names = set()

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue
        face_resized = cv2.resize(face, (160, 160))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_tensor = torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        try:
            embedding = model(face_tensor).detach().cpu().numpy()[0]
        except:
            continue

        name = "Unknown"
        min_dist = 1.0
        for known_emb, known_name in zip(known_embeddings, known_names):
            dist = np.linalg.norm(embedding - known_emb)
            if dist < 0.9 and dist < min_dist:
                min_dist = dist
                name = known_name

        if name != "Unknown" and name not in printed_names:
            status = mark_attendance(name)
            if status:
                printed_names.add(name)
                log_box.insert(tk.END, status + "\n")
                log_box.yview(tk.END)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
