import numpy as np
from PIL import Image
import os
import math
import cv2

def calculate_class_weights(folder_path, num_classes):
    class_counts = [0 for _ in range(num_classes)]
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".png"):
                label_path = os.path.join(root, filename)
                # label_image = Image.open(label_path)
                # label_array = np.array(label_image, dtype=np.uint8)
                label_array = cv2.imread(label_path,cv2.IMREAD_UNCHANGED)
                # pixels = np.array(image)
                label_array = cv2.split(label_array)
                label_array = label_array[2]
                
                unique_classes = np.unique(label_array)
                for cls in unique_classes:
                    class_counts[cls] += np.count_nonzero(label_array == cls)
    
    all_class_pixel_count = sum(class_counts)
    
    class_count_percent = [count / all_class_pixel_count for count in class_counts]
    
    median_percent = np.median(class_count_percent)
    middle_class = np.argmax(class_count_percent == median_percent)
    
    class_weight = [1 / math.log(count) if count != 0 else 0 for count in class_counts]
    
    weights = [40 * weight / sum(class_weight) for weight in class_weight]
    
    class_weights = {i: round(weights[i], 4) for i in range(num_classes)}
    
    return class_weights

# 文件夹路径和类别数
folder_path = "/home/ht/code/DELIVER/semantic/train"  # 替换为实际的文件夹路径
num_classes = 26  # 替换为实际的类别数

# 计算类别权重
class_weights = calculate_class_weights(folder_path, num_classes)

# 输出类别权重
class_weights_list = [class_weights[i] for i in range(num_classes)]
print("class_weights len:", len(class_weights_list))
print("class_weight = [")
for i, weight in enumerate(class_weights_list):
    print(weight, end="")
    if (i + 1) % 4 == 0 or i == num_classes - 1:
        print("," if i != num_classes - 1 else "")
    else:
        print(",", end=" ")
print("]")