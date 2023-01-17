# pip3 install kivy
# pip3 install numpy
# pip3 install opencv-python

# Eto na pinaka simple kong Code dun, kesa umabot ilang Line HAHAHA ayos na toh
# Ung Text to Speech na stop sya kada magsasalita, pero pede nmn yan ilagay sa ibang thread, dun sya magsalita
# Diko na nilagay antok nako.

from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.app import runTouchApp

# For Text to Speech, welp madame pang ibang version n2, ayos nato para standard
import win32com.client

# Pag android ginamit, pede gumamit ng plyer
# from plyer import tts
# How to use?:
# tts.speak("Something")

import numpy as np
import cv2

Window.size = (600, 600)
class MainObjectDetector(Widget):
    
    # Gumamit ako ng pre trained model
    # Para di nako gumamit ng Machine Learning
    # Alam ko meron na kau nun HAHA
    PrototxtPath = "Model.prototxt"
    ModelPath = "Model.caffemodel"
    MinimumConfidence = 0.5

    # Eto lang muna ung mga kaya i detect tas mahina pa HAHAHA, ayos na yan example lang nmn.
    # Yoko na maghanap pa ng ibang model HAHAHA
    CLASSES = ["background", "aeroplane", "bicycle",
           "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.detectPerSecond = 1
        self.framePerSecond = 30
        
        # Set Tha Random Seed Statically
        np.random.seed(543210)
        self._colors = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self._net = cv2.dnn.readNetFromCaffe(self.PrototxtPath, self.ModelPath)
        
        self.allDetectedObject = [0 for _ in range(len(self.CLASSES))]
        self.oldDetectedObject = self.allDetectedObject
        self.displayImage()
        self.startApplication()

    def convertCvtoImageTexture(self, image: cv2) -> Texture:
        """
            Convert CV Image to Texture
        """
        _image_bytes = image.tobytes()[::-1]
        _texture = Texture.create(size=(self._width_, self._height_), colorfmt="rgb")
        _texture.blit_buffer(_image_bytes, bufferfmt="ubyte", colorfmt="rgb")
        _texture.flip_horizontal()
        return _texture

    def startApplication(self) -> None:
        self._cap = cv2.VideoCapture(0)
        Clock.schedule_interval(lambda _: self.scanImage(), 1/self.framePerSecond)

    def scanImage(self) -> None:
        _, image = self._cap.read()
        self._height_, self._width_ = image.shape[0], image.shape[1]
        
        # Paglaruan nio kau bahala HAHAHA
        self._net.setInput(cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.005, (300, 300), 130))
        detected_objects = self._net.forward()
        
        self.allDetectedObject = [0 for _ in range(len(self.CLASSES))]
        for i in range(detected_objects.shape[2]):
            confidence = detected_objects[0, 0, i, 2]
            if confidence > self.MinimumConfidence:
                
                # Getting all the element in Detected Objects
                class_index  = int(detected_objects[0, 0, i, 1])
                upperLeftX   = int(detected_objects[0, 0, i, 3] * self._width_)
                upperLeftY   = int(detected_objects[0, 0, i, 4] * self._height_)
                lowerRightX  = int(detected_objects[0, 0, i, 5] * self._width_)
                lowerRightY  = int(detected_objects[0, 0, i, 6] * self._height_)  
                self.allDetectedObject[class_index] += 1
                PredictionText = f"{self.allDetectedObject[class_index]}: {self.CLASSES[class_index]}: {confidence:.2f}%"
                
                # Draw Triangle and Text in OpenCV2
                cv2.rectangle(image, (upperLeftX, upperLeftY), (lowerRightX, lowerRightY), self._colors[class_index], 3)
                cv2.putText(image, PredictionText, 
                            (upperLeftX, (upperLeftY - 15) if (upperLeftY > 30) else (upperLeftY + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,  self._colors[class_index], 2)

        # Every 0.5 Second check if all detected list have been changed.
        self.detectPerSecond -= 1 / self.framePerSecond
        if self.detectPerSecond <= 0:
            self.detectPerSecond = 0.5
            for index, num in enumerate(self.allDetectedObject):
                if num != self.oldDetectedObject[index] and num:
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    speaker.Speak(f"{num}: {self.CLASSES[index]}")
                    
            self.oldDetectedObject = self.allDetectedObject

        # Update Image Texture, Since para mapagana sa CP toh, ginawa ko nlng syang Texture
        # Cinonvert ko ung CV2 Image sa Bytes tapos ginamit ko pang Texture, di maganda pero atleast simple lang.
        self._image.size = (self._width_, self._height_)
        self._image.pos = (Window.width/2 - self._width_/2, Window.height/2-self._height_/2)
        self._image.texture = self.convertCvtoImageTexture(image)        

    def displayImage(self) -> None:
        self._image = Image(size=(300, 300), pos=(0, 0))
        self.add_widget(Label(text="Weak Camera Object Detector", pos=(Window.width/2-50, Window.height-100)))
        self.add_widget(self._image)

if __name__ == "__main__":
    runTouchApp(MainObjectDetector())
