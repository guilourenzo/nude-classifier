import tensorflow as tf
import numpy as np

model = tf.saved_model.load("model")

DEFAULT_FUNCTION_KEY = "serving_default"
inference_func = model.signatures[DEFAULT_FUNCTION_KEY]

image_size = 224

def predict_single_image(image_file):
    
    image_data = tf.keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size))

    # Convert the loaded image file to a numpy array
    image_array = tf.keras.preprocessing.image.img_to_array(image_data)
    image_array /= 255

    x_train = []
    x_train.append(image_array)
    x_test = np.array(x_train)

    predictions = inference_func(tf.constant(x_test))

    return predictions['dense_2'].numpy()[0][0], predictions['dense_2'].numpy()[0][1]
