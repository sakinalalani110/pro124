# To Capture Frame
import cv2

# To process image array
import numpy as np

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Load the pre-trained model
model = keras.models.load_model('/Users/sakinalalani/Desktop/ AI part in py/PRO-C110-Project-Boilerplate-main/converted_keras (1)/keras_model.h5')

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

    # Reading / Requesting a Frame from the Camera 
    status, frame = camera.read()

    # if we were successfully able to read the frame
    if status:

        # Flip the frame
        frame = cv2.flip(frame, 1)

        # Resize the frame
        resized_frame = cv2.resize(frame, (224, 224))  # Adjust the size as needed

        # Expand the dimensions
        input_data = np.expand_dims(resized_frame, axis=0)

        # Normalize the image before feeding to the model (assuming pixel values are in [0, 255])
        normalized_input = input_data / 255.0

        # Get predictions from the model
        predictions = model.predict(normalized_input)
        predicted_class = np.argmax(predictions[0])

        # Displaying the frames captured
        label = 'Rock' if predicted_class == 0 else 'Paper' if predicted_class == 1 else 'Scissors'
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('feed', frame)

        # Waiting for 1ms
        code = cv2.waitKey(1)

        # If space key is pressed, break the loop
        if code == 32:
            break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()
