target_area = 'white'
folder_path = 'media/Mateus_2021/' + target_area + '/'
result_folder = 'results/' + target_area + '/'
class_mask_extension = '_class.png'
hc_extension = '.hdr'

# public datasets
public_dataset_path = 'media/public/'
pavia_umat_path = public_dataset_path + 'PaviaU.mat'
pavia_umat_mask_path = public_dataset_path + 'PaviaU_gt.mat'
pavia_centre_umat_path = public_dataset_path + 'Pavia.mat'
pavia_centre_umat_mask_path = public_dataset_path + 'Pavia_gt.mat'
indian_pines_umat_path = public_dataset_path + 'Indian_pines_corrected.mat'
indian_pines_umat_mask_path = public_dataset_path + 'Indian_pines_gt.mat'
salinas_umat_path = public_dataset_path + 'Salinas_corrected.mat'
salinas_umat_mask_path = public_dataset_path + 'Salinas_gt.mat'
salinas_a_umat_path = public_dataset_path + 'SalinasA_corrected.mat'
salinas_a_umat_mask_path = public_dataset_path + 'SalinasA_gt.mat'

# Extensions
binary_extension = ".bin"
csv_extension = ".csv"
csv_delimiter = ","
numpy_binary_extension = ".npy"

# Train/Test split in system file
cube_extension = "_cube"
label_extension = "_label"

# Config
config_file = 'config.json'
