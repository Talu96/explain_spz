import cv2
import numpy as np

class Snake:
    def __init__(self, snake):
        self.snake =  snake
        self.area = self.compute_area()
        self.len = self.compute_len()
        self.roundness = self.compute_roundness()
        self.ratio = self.compute_ratio()

    def get_snake(self):
        return self.snake
        
    def get_area(self):
        return self.area
    
    def get_len(self):
        return self.len
    
    def get_roundness(self):
        return self.roundness
    
    def compute_area(self):
        x = self.snake[:, 1]
        y = self.snake[:, 0]
        return 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

    def compute_len(self):
        diff = np.diff(self.snake, axis=0)
        return np.sum(np.linalg.norm(diff, axis=1))

    def compute_roundness(self):
        return (4 * np.pi * self.area) / (self.len ** 2)

    def compute_ratio(self):
        if len(self.snake) < 5:  # fitEllipse richiede almeno 5 punti
            return None

        # reshape per OpenCV fitEllipse
        snake_reshaped = self.snake.reshape((-1, 1, 2)).astype(np.int32)

        ellipse = cv2.fitEllipse(snake_reshaped)
        (x, y), (MA, ma), angle = ellipse  # MA: asse maggiore, ma: asse minore

        if MA == 0:
            return 0

        return ma / MA

    def get_info_snake(self):
        return {
            "area": self.area,
            "lenght": self.len,
            "roundness": self.roundness,
            "ratio": self.ratio}

