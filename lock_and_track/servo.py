from gpiozero import Servo
from time import sleep

myGPIO = 25

myCorrection = 0.45
maxPW = (2.0 + myCorrection) / 1000
minPW = (1.0 - myCorrection) / 1000
midPW = (maxPW + minPW) / 2
servo = Servo(myGPIO, min_pulse_width=minPW, max_pulse_width=maxPW)
servo.value = -0.1


def rotate(val):
    servo.value = val
    sleep(0.01)
    servo.value = None


def stop():
    servo.value = None
