import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import json
import os

import numpy as np
import cairocffi as cairo
import matplotlib.pyplot as plt
import random
import json

with open('configuration.json', 'r') as config:
    location = json.load(config)

ndjson_data = location.get("raw ndjson data path")
raw_data = location.get("raw data path")


def vector_to_raster(vector_images, side=64, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):

    original_side = 256.

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()

        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1, 1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)

    return raster_images


def LoadNdjson(file_path):
    drawings = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            drawings.append(data['drawing'])
    return drawings

def SaveToNumpyFile(raster_images, file_name, path):
    file_name_without_ext = os.path.splitext(file_name)[0]
    np.save(os.path.join(path, file_name_without_ext + '.npy'), np.array(raster_images))

def ChangeFilesFormat(input_directory):
    directory = os.fsencode(input_directory)
    
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        file_path = input_directory + '/' + file_name
        drawings = LoadNdjson(file_path)
        raster_drawings = vector_to_raster(drawings)
        SaveToNumpyFile(raster_drawings, file_name, raw_data)

        print("file name:", file_name)

ChangeFilesFormat(ndjson_data)
