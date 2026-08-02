import os
import sys
import pdb
import json
import numpy as np
from PIL import Image,ImageDraw


shot_amount = "fewshot"
target_directory = os.path.join(".",f"{shot_amount}_mask")
#determine whether we're doing few shot or zero shot

if not os.path.exists(target_directory):
    os.makedirs(target_directory)

label_path = os.path.join(".",shot_amount,"labels.json")
print(label_path)
with open(label_path, "r") as json_file:
    data = json.load(json_file)


palette = [0, 0, 0, 204, 241, 227, 112, 142, 18, 254, 8, 23, 207, 149, 84, 202, 24, 214,
           230, 192, 37, 241, 80, 68, 74, 127, 0, 2, 81, 216, 24, 240, 129, 20, 215, 125, 161, 31, 204,
           254, 52, 116, 117, 198, 203, 4, 41, 68, 127, 252, 61, 21, 3, 142, 40, 10, 159, 241, 61, 36,
           14, 175, 77, 144, 61, 115, 131, 79, 97, 109, 177, 163, 58, 198, 140, 17, 235, 168, 47, 128, 91,
           238, 103, 45, 124, 35, 228, 101, 48, 232, 74, 124, 114, 78, 49, 30, 35, 167, 27, 137, 231, 47,
           235, 32, 39, 56, 112, 32, 62, 173, 79, 86, 44, 201, 77, 47, 217, 246, 223, 57, ]
palette = palette + [0]*(768-len(palette))
# pad it out so there's a full 256-color palette

imgs = list(data.keys)
for img in imgs:
    print(img) #debug print string
    img_path = os.path.join(".",shot_amount,img)
    image = Image.open(img_path).convert("RGB")
    (width,height) = image.size

    image_data = data[img]
    num_objects = len(image_data['regions'])
    mask = Image.new("L", (width,height),0)
    for idx_object in reversed(range(num_objects)):
        x_coords = image_data['regions'][str(idx_object)]['shape_attributes']['all_points_x']
        y_coords = image_data['regions'][str(idx_object)]['shape_attributes']['all_points_y']
        coords = [(x,y) for x,y in zip(x_coords, y_coords)]
        label = int(image_data['regions'][str(idx_object)]['region_attributes']['label'])
        print(len(coords), label) #debug print string
        ImageDraw.Draw(mask).polygon(coords, fill=label)
        print(idx_object, label)

    mask = np.array(mask)
    print(mask.shape, np.unique(mask))

    outpath = os.path.join(target_directory,img)
    processed_image = Image.fromarray(mask, 'P')
    processed_image.putpalette(palette)
    processed_image.save(outpath)