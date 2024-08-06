import os
import shutil

file_dir = 'low/'
for i in os.listdir(file_dir):
    origin_path = os.path.join(file_dir, i)
    target_path = i
    shutil.move(origin_path, target_path)