import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1) 

    return int(predicted_class[0])

def get_class_name(prediction):
    class_names = {
        0: 'Perro',
        1: 'Caballo',
        2: 'Elefante',
        3: 'Mariposa',
        4: 'Gallina'
    }
    return class_names.get(prediction, "Desconocido")





