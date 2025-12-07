# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
awsaf49_asvpoof_2019_dataset_path = kagglehub.dataset_download('awsaf49/asvpoof-2019-dataset')

print('Data source import complete.')
import os

src = os.path.join(awsaf49_asvpoof_2019_dataset_path, "LA", "LA")  # real folder
dst = "./data/LA"  # symlink to create in current directory

# Remove old symlink or folder if exists
if os.path.islink(dst) or os.path.exists(dst):
    os.remove(dst)

# Create symlink
os.symlink(src, dst, target_is_directory=True)

print("Symlink created:", dst, "→", src)