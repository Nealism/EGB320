'''Camera module for Raspberry Pi using Picamera2'''
from picamera2 import Picamera2

# Pi Camera settings
EXPOSURE = 10000
ANALOGUE_GAIN = 1.5
R_GAIN = 1.596
B_GAIN = 2.464

class Camera:
    '''Camera class for Raspberry Pi using Picamera2'''
    def __init__(self, resolution=(1280, 720)): #1280,720
        self.cap = Picamera2()
        config = self.cap.create_video_configuration(main={"format":'XRGB8888',"size":resolution})
        self.cap.configure(config)
        self.cap.set_controls({"ExposureTime": EXPOSURE,"AnalogueGain": ANALOGUE_GAIN,"AeEnable":False,"AwbEnable":False})
        self.cap.set_controls({"ColourGains": (R_GAIN, B_GAIN)})
        self.cap.start()

    def capture_frame(self):
        '''Capture a single frame from the camera'''
        frame = self.cap.capture_array()
        #a = self.cap.capture_metadata() 
        #print(a.get("ColourGains"))
        return frame

    def close(self):
        '''Release the camera resources'''
        self.cap.close()