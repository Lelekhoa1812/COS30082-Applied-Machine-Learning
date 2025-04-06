import cv2
import face_recognition
import os

# === Load known faces ===
known_face_encodings = []
known_face_names = []

# Iterate among files in KnowFaces directory
for filename in os.listdir("KnowFaces"):
    if filename.endswith(('.jpg', '.png')):
        img_path = os.path.join("known_faces", filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])  # Only take the first face
            known_face_names.append(os.path.splitext(filename)[0])
            print(f"INFO. Loaded: {filename}")
        else:
            print(f"WARNING. No face found in {filename}, skipping...")

# === Initialize webcam video ===
video_capture = cv2.VideoCapture(0)
recognized_count = 0

# Start streaming, ensure it's in RGB 
while True:
    print("INFO. Starting webcam. Press 'q' to quit...")
    ret, frame = video_capture.read()
    if not ret:
        print("ERROR. Failed to grab frame from webcam.")
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect all faces and encode them
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Detect face from encoded data and compare data
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Choose best match within heuristic distance threshold
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0 and min(face_distances) < 0.5:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Draw box and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 4, bottom - 4), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # Save snapshot image of recognized person (max 3)
        if name != "Unknown" and recognized_count < 3:
            img_name = f"recognized_{recognized_count+1}_{name}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"Saved: {img_name}")
            recognized_count += 1

    # Display the video frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q') or recognized_count >= 3:
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
print("INFO. Webcam stopped. Program ended.")