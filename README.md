# Face Embedding Generation

This script generates facial embeddings from a directory-based dataset used in a **face recognition system**.

## Overview

The script processes facial images and performs the following steps:

1. Loads the **YOLOv8** model for face detection.
2. Detects and validates a single face in each image.
3. Crops the detected face.
4. Uses **MTCNN** for face alignment.
5. Generates a **512-dimensional face embedding** using FaceNet (`InceptionResnetV1`).
6. Stores the embeddings for each student.
7. Saves all embeddings in a compressed `.npz` file.

## Technologies

* Python
* OpenCV
* NumPy
* PyTorch
* FaceNet (`facenet-pytorch`)
* MTCNN
* YOLOv8 (`Ultralytics`)

## Dataset Structure

The input dataset should be organized with one folder per person:

```text
Students/
├── Student_001/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
├── Student_002/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
└── Student_003/
    ├── image1.jpg
    ├── image2.jpg
    └── image3.jpg
```
Each student's folder name is used as their **student ID**.

## Output

The generated embeddings are saved as:
```text
student_embeddings_all_batch.npz
```

Each student's embeddings are stored as a NumPy array with the shape:
```text
(Number of images, 512)
```

## Usage

Update the dataset path in the script:
```python
directory_path = "Students"
```

Then run:
```bash
python embedding_generation.py
```
