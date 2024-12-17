from PIL import Image
import numpy as np
import os

image_dir = 'D:/PycharmProject/fetal/datafetal/labeled_data/labels' #D:/PycharmProject/fetal/datafetal/labeled_data/labels
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

image_arrays = []
for image_path in image_paths:
    image = Image.open(image_path)
    
    image_array = np.array(image)
    
    image_arrays.append(image_array)
print(f"image_array is {np.unique(image_arrays)} ")
 
label_pre_dir = 'D:/PycharmProject/fetal/datafetal/labeled_data/label_exchange'  

if not os.path.exists(label_pre_dir):
    os.makedirs(label_pre_dir)

for filename in os.listdir(image_dir):
    if filename.endswith('.png'):  
        image_path = os.path.join(image_dir, filename)
        
        image = Image.open(image_path)
        image_array = np.array(image)

        image_array[image_array == 1] = 128  
        image_array[image_array == 2] = 255  

        new_image = Image.fromarray(image_array.astype(np.uint8))  

        new_image_path = os.path.join(label_pre_dir, filename)
        new_image.save(new_image_path)

print("已经另存到label_exchange中")
