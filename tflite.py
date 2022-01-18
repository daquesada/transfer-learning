import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

img, x = get_image('aguamanil.jpg')
x=x.astype('float32') / 255.

#Load model
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#load image
input_shape = input_details[0]['shape']
input_data = x
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

df = pd.DataFrame(data=output_data,columns=['d543', 'aguamanil-c716', 'c717', 'd575-caimito', 'a21-Palialte-Amarillo', 'd574-cargadero', 'sajo'])

print("{}".format(df))