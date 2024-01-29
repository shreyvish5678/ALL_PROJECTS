# Import libraries
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import openai
# Load pre-trained model
model = tf.keras.models.load_model('./road_sign.h5')
# Define class dictionary for signs
class_dict = {0: 'Speed limit (20km/h)',
              1: 'Speed limit (30km/h)',
              2: 'Speed limit (50km/h)',
              3: 'Speed limit (60km/h)',
              4: 'Speed limit (70km/h)',
              5: 'Speed limit (80km/h)',
              6: 'End of speed limit (80km/h)',
              7: 'Speed limit (100km/h)',
              8: 'Speed limit (120km/h)',
              9: 'No passing',
              10: 'No passing for vehicles over 3.5 metric tons',
              11: 'Right-of-way at the next intersection',
              12: 'Priority road',
              13: 'Yield',
              14: 'Stop',
              15: 'No vehicles',
              16: 'Vehicles over 3.5 metric tons prohibited',
              17: 'No entry',
              18: 'General caution',
              19: 'Dangerous curve to the left',
              20: 'Dangerous curve to the right',
              21: 'Double curve',
              22: 'Bumpy road',
              23: 'Slippery road',
              24: 'Road narrows on the right',
              25: 'Road work',
              26: 'Traffic signals',
              27: 'Pedestrians',
              28: 'Children crossing',
              29: 'Bicycles crossing',
              30: 'Beware of ice/snow',
              31: 'Wild animals crossing',
              32: 'End of all speed and passing limits',
              33: 'Turn right ahead',
              34: 'Turn left ahead',
              35: 'Ahead only',
              36: 'Go straight or right',
              37: 'Go straight or left',
              38: 'Keep right',
              39: 'Keep left',
              40: 'Roundabout mandatory',
              41: 'End of no passing',
              42: 'End of no passing by vehicles over 3.5 metric tons'}
# Load the image and convert to grayscale
img = cv2.imread('input_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Resize the image to 32x32 pixels
resized = cv2.resize(gray, (32,32), interpolation = cv2.INTER_AREA)
# Convert the image to PIL format
pil_image = Image.fromarray(resized)
# Preprocess the image
img_norm = (np.sum(np.expand_dims(np.array(pil_image), axis=-1)/3, axis=-1, keepdims=True) - 128) / 128
# Make a prediction
prediction = model.predict(img_norm.reshape(1, 32, 32, 1))
# Get the predicted class
predicted_class = class_dict[np.argmax(prediction)]
print(predicted_class)
# Use ChatGPT to generate a response about the sign
openai.api_key = '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' 
# Message generation using ChatCompletion
message = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant that gives users a description of a given road sign, and recommendations on what to do when they see it."},
        {"role": "user", "content": "What  should I do if I see a 65 mph sign"},
        {"role": "assistant", "content": "Keep your speed below 65 miles per  hour, as it will keep you safe."},
        {"role": "user", "content": "Tell me about this sign: " + predicted_class},
    ]
)
#Extract the output from json file
output = message.get('choices')[0]
content = output.get('message')
response = content.get('content') 
print('This is the "' + predicted_class + '" sign. '  + response)
