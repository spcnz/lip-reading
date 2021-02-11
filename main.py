import os
import glob

DATASET_RANGE = list(range(1, 21)) + list(range(22, 35))
print (DATASET_RANGE)
def get_paths(speakers, data_type):
    #glob_data = os.path.join('features', '*', '.npy')
    file_paths = []
    for sp in speakers:
        glob_data = os.path.join(data_type, 's'+str(sp), '*.npy')
        file_paths.extend(glob.glob(glob_data))       
    return file_paths
feature_paths  = get_paths(DATASET_RANGE, 'features')
print(feature_paths)
