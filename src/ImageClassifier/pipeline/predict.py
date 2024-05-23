import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    
    def predict(self):
        model = load_model(os.path.join("artifacts","training","model.h5"))
        
        imagename = self.filename
        test_image = image.load_img(imagename, target_size= (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)/255
        prob = model.predict(test_image)
        result = np.argmax(model.predict(test_image), axis = 1)
        
        class_names = ["Angelina Jolie", "Brad Pitt", "Denzel Washington", "Hugh Jackman", "Jennifer Lawrence", "Johnny Depp",
                       "Kate Winslet", "Leonardo DiCaprio", "Megan Fox", "Natalia Portman",
                       "Nicole Kidman", "Robert Downey Jr", "Sandra Bullock", "Scarlett Johansson", "Tom Cruise", "Tom Hanks", "Will Smith"]
        # print(prob)
        print(result)
        # print(class_names[result[0]])
        return [class_names[result[0]]]

if __name__ == "__main__":
    filename = "C:\\Users\\digvi\\OneDrive\\Documents\\Image_classification\\artifacts\\data_ingestion\\Celebrity Faces Dataset\\Will Smith\\001_beebcee2.jpg"
    obj = PredictionPipeline(filename=filename)
    obj.predict()