# import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image

# Function to load and encode faces
def load_encoded_faces():
    # Load images and encode faces
    # You need to have these images in your project directory or specify the correct path
    sachin_img = face_recognition.load_image_file("faces/sachin.jpg")
    sachin_encoding = face_recognition.face_encodings(sachin_img)[0]

    raj_img = face_recognition.load_image_file("faces/raj.jpg")
    raj_encoding = face_recognition.face_encodings(raj_img)[0]

    minal_img = face_recognition.load_image_file("faces/minal.jpg")
    minal_encoding = face_recognition.face_encodings(minal_img)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [sachin_encoding, raj_encoding, minal_encoding]
    known_face_names = ["Sachin", "Raj", "Minal"]
    return known_face_encodings, known_face_names

# Function to recognize faces in an image
def recognize_faces(input_image, known_face_encodings, known_face_names):
    # Convert the image to RGB
    rgb_image = input_image.convert('RGB')
    image_np = np.array(rgb_image)

    # Find all face locations and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        names.append(name)

    return face_locations, names

def main():
    st.title("Face Recognition Attendance System")

    known_face_encodings, known_face_names = load_encoded_faces()

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Recognize Faces"):
            face_locations, names = recognize_faces(image, known_face_encodings, known_face_names)

            # Draw rectangles around faces
            draw = ImageDraw.Draw(image)
            for (top, right, bottom, left), name in zip(face_locations, names):
                # Draw a box around the face
                draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

                # Draw a label with a name below the face
                text_width, text_height = draw.textsize(name)
                draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

            st.image(image, caption="Recognized Faces", use_column_width=True)

if __name__ == "__main__":
    main()
