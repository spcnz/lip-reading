import os
import glob
import shutil

from constants import FEATURE_DIRECTORY_PATH

def group_speaker(sp, test_p, train_p):
    glob_data = os.path.join(FEATURE_DIRECTORY_PATH, 's'+str(sp), '*.npy')
    files = glob.glob(glob_data)
    count = len(files)
    train_files_num = train_p * count // 100
    test_files_num = test_p * count // 100
    test = files[:test_files_num]
    train = files[test_files_num:count]


    test_path = os.path.join(FEATURE_DIRECTORY_PATH,'s'+str(sp), 'test\\')
    print(test_path)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    for p in test:
        file_name = p.split('\\')[-1]
        print(file_name)
        shutil.copy(p, os.path.join(test_path, file_name))

    train_path = os.path.join(FEATURE_DIRECTORY_PATH,'s'+str(sp), 'train\\')
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    for p in train:
        file_name = p.split('\\')[-1]
        print(file_name)
        shutil.copy(p, os.path.join(train_path, file_name))

for i in range(2,35):
    if (i == 21):
        continue
    group_speaker(i, 15, 85)