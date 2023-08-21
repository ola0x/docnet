import os
import numpy as np

from keras import Model
from keras import Sequential
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation

class DeepModel():
    '''DocNet deep model.'''
    def __init__(self):
        self._model = self._define_model()
        self._findCosineDistance = self.findCosineDistance()

        print('Loading DocNet.')
        print()
    
    @staticmethod
    def _define_model():
        global model
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

        model.load_weights('docnet.h5')

        model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

        return model
    
    @staticmethod
    def findCosineDistance(source_representation, test_representation):
        '''Calculating the distance of two inputs.

        The return values lies in [-1, 1]. `-1` denotes two features are the most unlike,
        `1` denotes they are the most similar.

        Args:
            input1, input2: two input numpy arrays.

        Returns:
            Element-wise cosine distances of two inputs.
        '''
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
    @staticmethod
    def findEuclideanDistance(source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance
    
    @staticmethod
    def process_img(image_path):
        '''Process an image to numpy array.

        Args:
            path: the path of the image.

        Returns:
            Numpy array of the image.
        '''
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x
    
    @staticmethod
    def find_similar_images(self, image_representation, db_representations, threshold=0.06):
        similarities = [self._findCosineDistance(image_representation, db_repr) for db_repr in db_representations]
        similar_indices = [i for i, similarity in enumerate(similarities) if similarity < threshold]
        return similar_indices
    
    def extract_feature(self, representations):
        '''Extract deep feature using MobileNet model.

        Args:
            generator: a predict generator inherit from `keras.utils.Sequence`.

        Returns:
            The output features of all inputs.
        '''
        features = self._model.vgg_face_descriptor.predict(representations)[0,:]
        return features