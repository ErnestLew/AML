from keras.models import load_model
import cv2
import numpy as np
import streamlit as st
import tempfile

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 15
CLASSES_LIST = ['moving_arms', 'hand_shaking', 'nothing']

# Load the trained model
model = load_model('LRCN.h5')

def process_video(uploaded_file, sequence_length=SEQUENCE_LENGTH, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH):
    # Save the uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())

    video = cv2.VideoCapture(tfile.name)
    frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (image_width, image_height))
        normalized_frame = resized_frame / 255.0
        frames.append(normalized_frame)

        if len(frames) == sequence_length:
            yield np.array([frames])
            frames = []

    video.release()
    tfile.close()  # Close and delete the temporary file

def process_and_predict(model, uploaded_file, sequence_length=SEQUENCE_LENGTH):
    predictions = []
    second = 0
    results = []

    for frames in process_video(uploaded_file, sequence_length):
        prediction = model.predict(frames)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = CLASSES_LIST[predicted_class_index]
        predictions.append(predicted_class_index)
        results.append(f"Second {second}: Predicted action - {predicted_class}")
        second += 1

    unique, counts = np.unique(predictions, return_counts=True)
    prediction_counts = dict(zip(unique, counts))
    prediction_counts.pop(2, None)  # Remove 'nothing' class

    if prediction_counts:
        most_common_class_index = max(prediction_counts, key=prediction_counts.get)
        final_prediction = CLASSES_LIST[most_common_class_index]
    else:
        final_prediction = "This video has no specified actions"

    return final_prediction, results

st.set_page_config(page_title='Fruit Classifier', page_icon='icons\icon.png', layout='centered')
st.title("Video Classification App")
st.write("This app classifies the actions in a video.")
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    final_prediction, prediction_results = process_and_predict(model, uploaded_file)
    st.write("Final Prediction for the video: ", final_prediction)