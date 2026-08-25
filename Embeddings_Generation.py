import os
import cv2
import numpy as np
import torch
import gc  # Garbage Collector
from facenet_pytorch import MTCNN, InceptionResnetV1  # type: ignore
from ultralytics import YOLO

# Initialize YOLOv8 model for face detection
model = YOLO("yolov8m-face.pt")
print("✅ YOLOv8 model loaded successfully.")

# Initialize MTCNN and FaceNet model
mtcnn = MTCNN(keep_all=False)  # Detect only one face per image
resnet = InceptionResnetV1(pretrained='vggface2').eval()
print("✅ MTCNN and InceptionResnetV1 models loaded successfully.")


def save_embeddings_from_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"❌ Error: '{directory_path}' is not a valid directory.")
        return

    all_embeddings = {}  # student_id -> list of embeddings
    total_students = len(os.listdir(directory_path))
    print(f"📚 Total students found: {total_students}")

    student_counter = 0
    image_counter = 0

    for student_folder in sorted(os.listdir(directory_path)):
        person_path = os.path.join(directory_path, student_folder)
        if not os.path.isdir(person_path):
            continue  # Skip non-folder files

        student_counter += 1
        student_id = student_folder
        person_embeddings = []
        print(f"\n👤 Processing Student {student_counter}/{total_students}: '{student_id}'")

        for filename in sorted(os.listdir(person_path)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue  # Skip unsupported file formats

            image_path = os.path.join(person_path, filename)

            # --- Load image safely ---
            try:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not read image {image_path}")
            except Exception as e:
                print(f"⚠️ Skipping '{filename}': {e}")
                continue

            # Resize if too large (optional, speeds up processing)
            if img.shape[0] > 1000 or img.shape[1] > 1000:
                img = cv2.resize(img, (160, 160))

            image_counter += 1
            print(f"📸 Processing image {image_counter}: '{filename}' for ID '{student_id}'...")

            # --- Detect face with YOLO ---
            results = model(img)
            num_faces = len(results[0].boxes)

            if num_faces == 0:
                print(f"❌ No face detected in '{filename}'. Skipping...")
                del img, results
                gc.collect()
                continue
            if num_faces > 1:
                print(f"⚠️ Multiple faces detected in '{filename}'. Skipping...")
                del img, results
                gc.collect()
                continue

            # Extract bounding box
            box = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            # Crop and validate face region
            face = img[y1:y2, x1:x2]
            if face.shape[0] == 0 or face.shape[1] == 0:
                print(f"⚠️ Invalid face region in '{filename}'. Skipping...")
                del img, face, results
                gc.collect()
                continue

            # Convert cropped face to RGB
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Align face with MTCNN
            face_tensor = mtcnn(face_rgb)
            if face_tensor is None:
                print(f"⚠️ MTCNN could not find a face in '{filename}'. Skipping...")
                del img, face, face_rgb, results
                gc.collect()
                continue

            # --- Generate face embeddings ---
            with torch.no_grad():
                embedding = resnet(face_tensor.unsqueeze(0)).cpu().numpy().flatten().astype(np.float32)
            person_embeddings.append(embedding)
            print(f"✅ Embedding generated for '{filename}'.")

            # --- Free memory after each image ---
            del img, face, face_rgb, results, face_tensor, embedding
            gc.collect()

        # Save embeddings for this student
        if person_embeddings:
            all_embeddings[student_id] = np.vstack(person_embeddings)  # (N, 512)

    # Save all embeddings to compressed .npz
    if all_embeddings:
        np.savez_compressed('student_embeddings_all_batch.npz', **all_embeddings)
        print("\n✅ All embeddings saved successfully to 'student_embeddings.npz'.")
    else:
        print("❌ No embeddings were generated.")


# Example usage
directory_path = "Students"  # <-- change this to your dataset path
save_embeddings_from_directory(directory_path)
