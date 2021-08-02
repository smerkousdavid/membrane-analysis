import json
import os
import re
from typing import List

import htmlmin
from jsmin import jsmin as jsminify
from rcssmin import cssmin
from structure.interactive.base import ResultLayer

# use the following template
FOLDER_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FILE = os.path.join(FOLDER_DIR, 'template.html')
MAX_REPLACE = 2


class ImageExport(object):
    def __init__(self, temp_file=TEMPLATE_FILE):
        self.template = None
        self.load_template(temp_file)
    
    def load_template(self, temp_file=TEMPLATE_FILE):
        with open(temp_file, 'r') as tfile:
            self.template = tfile.read()
        
        # minify the template
        self.template = htmlmin.Minifier().minify(self.template)
    
        # convert and minify the javascript
        match = re.search(r'<script.*>((.|\n)*)</script>', self.template, re.M)
        if match:
            javascript = match.group(1)
            minified = jsminify(javascript)
            self.template = self.template.replace(javascript, minified)

        # convert and minify the css
        match = re.search(r'<style.*>((.|\n)*)</style>', self.template, re.M)
        if match:
            css = match.group(1)
            minified = cssmin(css)
            self.template = self.template.replace(css, minified)

    def __make_export(self, data: dict):
        export = self.template
        for key in data.keys():
            export = export.replace(key, str(data[key]), MAX_REPLACE)
        return export

    def get_export_data(self, image_name: str, results: List[ResultLayer]):
        if self.template is None:
            raise ValueError('Template file is empty or was not properly loaded')
        
        # get largest dims and construct data
        large_width = 0
        large_height = 0
        all_data = {}
        layers = []
        for res in results:
            if res.image and res.image is not None and res.dims is not None:
                if res.dims[0] > large_width:
                    large_width = res.dims[0]
                if res.dims[1] > large_height:
                    large_height = res.dims[1]
            
            # add the data and the layer information
            res.finalize()
            all_data[res.name] = res.ref_data
            all_data.update(res.global_data)
            layers.append(res.get_def())

        # set the image dimensions in the template
        export = self.__make_export({
            '%%IMAGE_WIDTH%%': large_width,
            '%%IMAGE_HEIGHT%%': large_height,
            '%%ALL_DATA%%': json.dumps(all_data, separators=(',', ':')),
            '%%IMAGE_LAYERS%%': json.dumps(layers, separators=(',', ':')),
            '%%IMAGE_NAME%%': image_name
        })

        return export

    def write_export(self, path: str, image_name: str, results: List[ResultLayer]):
        with open(path, 'w') as imwrite:
            imwrite.write(self.get_export_data(image_name, results))

    def __parse_data(self, data: str, regex: str, group: int=1, js_data: bool=False) -> str:
        if data is None or regex is None:
            raise ValueError('Neither data nor regex can be none')
        
        found = re.search(regex, data)
        if found is None:
            raise ValueError('The pattern ' + regex + ' was not found in exported data')
        
        data = found.group(group)
        if data is None:
            raise ValueError('The group %d was not found in exported data' % group)

        if js_data:
            return json.loads(data)
        return data

    def load_import(self, data: str):
        """ @Deprecated look at dataset.py to see the new method of handling all non-basic export data """ 
        all_data = self.__parse_data(data, 'ALL_DATA\=JSON\.parse\(\'({.*})\'\);', js_data=True)
        layer_data = self.__parse_data(data, 'IMAGE_LAYERS\=JSON\.parse\(\'({.*\})\'\);', js_data=True)
