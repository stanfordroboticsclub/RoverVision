import numpy as np

class camUnit:

    def __init__(self, res_width, aov, obj_width, angle_offset, distance_offset):
        self.res_width = res_width #number of pixels of taken picture
        self.aov = aov #angle of view of the camera
        self.obj_width = obj_width #true width of the object
        self.angle_offset = angle_offset #angle of the camera respect to forward
        self.distance_offset = distance_offset #distance of camera to the center

    def has_obj(self):
        return False

    def get_pos(self):
        return 0

    def get_pixel_width(self):
        return 0

    def get_obj_distance(self):
        width_of_field = self.obj_width / self.get_pixel_width() * self.res_width
        return width_of_field / (2 * np.tan(np.pi * self.aov/360))

    def get_obj_angle(self):
        return np.arctan(get_pos() / self.get_obj_distance())

    def get_distance(self):
        return np.sqrt(((self.distance_offset + self.get_obj_distance()) ** 2) + (self.get_pos()**2))

    def get_heading(self):
        angle = np.arctan(self.get_pos() / (self.distance_offset + self.get_obj_distance()))
        return self.angle_offset + angle
