""" Handles base objects for drawing """
import base64
import json

import cv2
import numpy as np

# javascript evals
FUNCTION_BASE = '(ctx, data) => {'
FUNCTION_END = '};'
ALL_DATA_REF = 'data'


class ResultLayer(object):
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name
        self.image = False
        self.show = True
        self.data = None
        self.alpha = 1.0
        self.dims = None
        self.ref_data = None
        self.global_data = {}
    
    def set_reference_data(self, reference):
        self.ref_data = reference

    def add_global_data(self, key, data):
        self.global_data[key] = data

    def draw_image(self, image: np.ndarray, dimesions: tuple):
        self.image = True
        self.dims = dimesions  # we want to compare all of the dimensions later on

        # make sure there is some sort of image data
        failed = image is None
        if not failed:
            failed = len(image.shape) < 2 or len(image.shape) > 3

        _, arr = cv2.imencode('.jpg', image)
        self.data = 'data:image/jpeg;base64,' + base64.b64encode(arr.tobytes()).decode('utf-8')

    def __ensure_func(self):
        if self.data is None:
            self.data = FUNCTION_BASE

    def rgb(self, rgb: tuple):
        return 'rgb({}, {}, {})'.format(int(rgb[2]), int(rgb[1]), int(rgb[0]))

    def index(self, item: str):
        return '[\\"' + item + '\\"]'
 
    def get_path(self, path: str):
        return ALL_DATA_REF + self.index(self.name) + path

    def get_global_path(self, path: str):
        return ALL_DATA_REF + path

    def write_line(self, line: str):
        self.__ensure_func()
        self.data += line + ';'

    # def draw_line(self, path: str, close: bool=False, stroke: str='black', width: int=10, corner: str='round'):
    #     self.write_line('drawPolyLine(ctx, {}, {}, s={}, w={}, c={})'.format(path, 'true' if close else 'false', stroke, width, corner))

    def draw_line(self, start: tuple, end: tuple, stroke: str='black', width: int=10, corner: str='round'):
        self.write_line('drawLine(ctx,{},{},s=\\"{}\\",w={},c=\\"{}\\")'.format('[{},{}]'.format(start[0], start[1]), '[{},{}]'.format(end[0], end[1]), stroke, width, corner))

    def draw_poly_line(self, path: str, close: bool=False, stroke: str='black', width: int=10, corner: str='round'):
        self.write_line('drawPolyLine(ctx,{},{},s=\\"{}\\",w={},c=\\"{}\\")'.format(path, 'true' if close else 'false', stroke, width, corner))

    def draw_circle(self, data: (tuple, str), radius: (str, int), fill: str='black'):
        self.write_line('drawCircle(ctx,{},{},f=\\"{}\\")'.format(('[{},{}]'.format(data[0], data[1])) if isinstance(data, (tuple, list)) else data, radius, fill))

    def draw_text(self, text: str, x: int, y: int, color: str='black', font='18px Arial'):
        self.write_line('drawText(ctx,{},{},{},s=\\"{}\\",f=\\"{}\\")'.format(text, x, y, color, font))

    def draw_rect(self, x: int, y: int, width: int, height: int, color: str='black'):
        self.write_line('drawRect(ctx,{},{},{},{},f=\\"{}\\")'.format(x, y, width, height, color))

    def finalize(self):
        if not self.image and self.data is not None:
            if not self.data.endswith(FUNCTION_END):
                self.data += FUNCTION_END  # when drawing regular objects add the end of the call function
    
    def get_def(self):
        return {
            "id": self.id,
            "name": self.name,
            "image": self.image,
            "show": self.show,
            "data": self.data,
            "alpha": self.alpha
        }
