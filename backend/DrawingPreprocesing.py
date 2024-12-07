import numpy as np
import json
import cairocffi as cairo
import matplotlib.pyplot as plt



#Align the drawing to the top-left corner, to have minimum values of 0

def align_to_top_left(drawing_data):
    all_x = [x for stroke in drawing_data for x in stroke[0]]
    all_y = [y for stroke in drawing_data for y in stroke[1]]

    x_min = min(all_x)
    y_min = min(all_y)

    aligned = []
    strokes = []

    for stroke in drawing_data:
            
        X = []
        Y = []

        for x in stroke[0]:
            x = x - x_min
            X.append(x)
        for y in stroke[1]:
            y = y - y_min
            Y.append(y)
        strokes = (X,Y)
        aligned.append(strokes)

    return aligned


#Uniformly scale the drawing, to have a maximum value of 255

def scale_drawing(data):
    all_x = [x for stroke in data for x in stroke[0]]
    all_y = [y for stroke in data for y in stroke[1]]

    x_max = max(all_x)
    y_max = max(all_y)

    scale_x = 255/x_max
    scale_y = 255/y_max

    print(scale_x)
    print(scale_y)

    scaled = []
    strokes = []

    for stroke in data:
            
        X = []
        Y = []

        for x in stroke[0]:
            x = round(x * scale_x)
            X.append(x)
        for y in stroke[1]:
            y = round(y * scale_y)
            Y.append(y)
        strokes = (X,Y)
        scaled.append(strokes)

    return scaled

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

def display_raster_images(raster_images, side):
    if len(raster_images) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(raster_images[0].reshape((side, side)), cmap='gray')
        ax.axis('off')
    else:
        fig, axes = plt.subplots(1, len(raster_images), figsize=(10 * len(raster_images), 10))
        for ax, raster_image in zip(axes, raster_images):
            ax.imshow(raster_image.reshape((side, side)), cmap='gray')
            ax.axis('off')  # Hide axes
    plt.show()

def save_to_npy(raster_images, output_file):
    raster_images = [drawing.reshape(64, 64).astype(np.float32) / 255.0 for drawing in raster_images]

    raster_images = [np.squeeze(drawing) for drawing in raster_images]

    np.save(output_file, np.array(raster_images))
    print(f"Npy drawing saved to {output_file}")

def load_drawing_data(input_file):
    with open(input_file, 'r') as f:
        drawing_data = json.load(f)
    return drawing_data

"""
input_file = 'drawing15.json'
output_file = 'img10'

   
drawing_data = load_drawing_data(input_file)


aligned = align_to_top_left(drawing_data)
scaled_drawing = scale_drawing(aligned)
raster_img = vector_to_raster([scaled_drawing])

save_to_npy(raster_img, output_file)

"""
