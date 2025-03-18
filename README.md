# GAMBAS

### Preprocessing

Before training the model, register input images to corresponding target images and ensure files are arranged in the following format:

	├── Training_data                   
	|   ├── images               
	|   |   ├── 0.nii 
  	|   |   ├── 1.nii 
	|   |   └── 2.nii                   
	|   ├── labels                       
	|   |   ├── 0.nii 
  	|   |   ├── 1.nii 
	|   |   └── 2.nii 

 ### Training

Modify "BaseOptions.py" to set directory for preprocessed training data (--data path) and validation data (--val_path). Select model that will be used for training by modifying --model (e.g. gambas, cycle_gan, ea_gan) and make sure correct patch size is specified via --patch size. Finally, set checkpoint directory (--checkpoints_dir) and project name (--name).

For standard training script use train.py, however to make sure TensorBoard log is created, use train_TB.py
    
