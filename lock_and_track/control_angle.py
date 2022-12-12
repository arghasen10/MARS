from time import sleep
import magnetometer
import servo


class ServoController:
    def __init__(self):
        self.theta_mm = magnetometer.get_magnetometer_reading()
        self.rotate_ccw = 0.0001
        self.rotate_cw = - 0.17
        self.epsilon = 2

    def rotate(self, aoa):
        self._rotate(aoa)
        self.theta_mm = magnetometer.get_magnetometer_reading()

    def _rotate(self, aoa):
        angle_val = magnetometer.get_magnetometer_reading()
        destination_angle = angle_val + aoa
        print('Starting angle', angle_val)
        if 10 > angle_val or angle_val > 340:
            servo.stop()
            return
        while abs(angle_val - destination_angle) > self.epsilon:
            angle_val = magnetometer.get_magnetometer_reading()
            if angle_val > destination_angle:
                servo.rotate(self.rotate_ccw)
            elif angle_val < destination_angle:
                servo.rotate(self.rotate_cw)
        servo.stop()
        print('Ending angle', angle_val)
