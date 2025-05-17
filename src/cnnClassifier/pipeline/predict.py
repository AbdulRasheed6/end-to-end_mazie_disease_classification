import numpy as np
import tf_keras as tf
from tf_keras.models import load_model
from tf_keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self, filename):
        self.filename= filename


    
    def predict(self):
        #load model
        model= load_model(os.path.join("artifacts", "training", "model.h5"))

        imagename= self.filename
        test_image= image.load_img(imagename, target_size= (224,224))
        test_image= image.img_to_array(test_image)
        test_image= test_image / 255.0
        test_image= np.expand_dims(test_image, axis=0)
        result= np.argmax(model.predict(test_image), axis=1)[0]
        print(result)

        class_names = ['HEALTHY', 'MLN', 'MSV']
        prediction = class_names[result]
        return [{"image": prediction}]



"""if result[0] ==0:
            prediction= 'HEALTHY'
            return [{"image": prediction}]
        
        if result[0]== 1:
            prediction= 'MLN'
            return [{"image": prediction}]

        else:
            prediction= 'MSV'
            return [{"image": prediction}]"""