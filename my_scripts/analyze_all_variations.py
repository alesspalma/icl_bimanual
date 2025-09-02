import os
import pickle

path = '/home/alessio/Desktop/icl_bimanual/generated_data/train/bimanual_dual_push_buttons/all_variations/episodes'
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder) # episode folder
    pkl_file = os.path.join(folder_path, 'variation_number.pkl')
    with open(pkl_file, 'rb') as f:
        variation_number = pickle.load(f)
    print(f"Folder: {folder}, Variation: {variation_number}")
print("Total number of episodes:", len(os.listdir(path)))