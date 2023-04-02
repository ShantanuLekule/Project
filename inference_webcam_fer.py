from config import *
#from Dataloader import *
from model import *
#from train import *ghts)
model_fer = load_model(weights)
model_fer.load_weights(weights)

#resDir = r''
casDir = r'C:\Users\shant\Downloads\common_project (1)\common_projecthaarcascades'

# program to capture single image from webcam in python

# importing OpenCV library
import cv2
import matplotlib.pyplot as plt

# initialize the camera
# If you have multiple camera connected with
# current device, assign a value in cam_port
# variable according to that
cam_port = 0
cam = cv2.VideoCapture(cam_port)

# reading the input using the camera


#os.chdir(inferenceDir)

faceCascade=cv2.CascadeClassifier(os.path.join(casDir,r'C:\Users\shant\Downloads\common_project (1)\common_projecthaarcascades\haarcascade_frontalface_default.xml'))  


videoCapture = cv2.VideoCapture(0)

# If image will detected without any error,
# show result
while True:
    _,image=videoCapture.read()
    faces = faceCascade.detectMultiScale(image,1.25,10)
    
    for x,y,w,h in faces:
        RGB_BLUE = (0,0,255)
        cv2.rectangle(image ,(x,y),(x+w,y+h),RGB_BLUE,3)
        print(image.shape)
        face = image[y:y+h, x:x+w, :]
        print(face.shape)
        face = cv2.resize(face, (224, 224))
        A=model_fer.predict(np.array([face]))
        predicted_class = np.argmax(A[0])
        fer= ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        z=fer[predicted_class]
        cv2.putText(image, z, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 0, 0), 2, cv2.LINE_AA)
        print(predicted_class)
        cv2.imshow('image',image)
        if predicted_class == 1 :
            os.chdir(HappySongDir)

            files=os.listdir()
            song = random.choice(files)        
            
            print('\n\nplaying happy songs\n\n')
            #print(f'list of songs\n{files}\n\n') 
            import pygame
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.music.load(song)
            pygame.mixer.music.play()

        
            print(f'currently playing :- {song}')
            
        
        elif predicted_class == 0 :
            
            os.chdir(SadSongDir)
        
            
            files=os.listdir()
            song = random.choice(files)        
            
            print('\n\nplaying sad songs\n\n')
            #qprint(f'list of songs\n{files}\n\n')
            
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.music.load(song)
            pygame.mixer.music.play()
            
            print(f'currently playing :- {song}')

            import time
            time.sleep(10)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break




cam.release() 
cv2.destroyWindow("photo")

cv2.destroyAllWindows()

