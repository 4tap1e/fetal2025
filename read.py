import os
from PIL import Image

def get_image_sizes(label_dir):
    # 获取文件夹下所有PNG文件的路径
    png_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]
    
    sizes = {}  # 用于存储每个图片的尺寸信息

    for file in png_files:
        file_path = os.path.join(label_dir, file)
        # 打开图片并获取其尺寸信息
        with Image.open(file_path) as img:
            width, height = img.size  # 获取宽度和高度
            sizes[file] = (width, height)
    
    return sizes

# 示例调用
label_dir = 'D:/PycharmProject/fetal/datafetal/labeled_data/labels'  # label文件夹路径
image_sizes = get_image_sizes(label_dir)

# 打印所有PNG文件的尺寸信息
for file, size in image_sizes.items():
    print(f"Image: {file}, Size: {size}") 
    #(544, 336)


