# MRI-GAN-suite

### Preprocessing

Before training the model, set data_dir in "Base_options.py" and run "run preproc.py" to convert data into h5 file format (required format for the dataloader). Files should be arranged in the following format prior to running the preprocessing script:

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

Modify "BaseOptions.py" to set directory for preprocessed training data (--data path) and validation data (--val_path). Select model that will be used for training by modifying --model (e.g. pix2pix, cycle_gan, ea_gan) and make sure correct patch size is specified via --patch size. Finally, set checkpoint directory (--checkpoints_dir) and project name (--name).

For standard training script use train.py, however to make sure TensorBoard log is created, use train_TB.py
    
