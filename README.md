<hr>
<h1 align="center">
  GAMBAS <br>
  <sub>Generalised-Hilbert Mamba for Super-resolution of Paediatric Ultra-Low-Field MRI</sub>
</h1>


<h3 align="center">[<a href="https://arxiv.org/abs/2504.04523">arXiv</a>]</h3>

Official PyTorch implementation of **GAMBAS**, a Mamba infused adversarial model, leveraging a generalised-Hilbert scan to process 3D medical imaging data with state-space models (SSMs).

<img src="GAMBAS_architecture.jpg" width="800px"/>

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

For standard training use train.py, or to integrate TensorBoard logging use train_TB.py


### Prediction

Once the model is trained, modify "TestOptions.py" to specify input image (via --image) and output name (via --result), then run "test.py" to obtain prediction.

### Pre-trained model

If using our-pretrained model, make sure to rigidly register yout input image to the template provided (Template.nii.gz). This will ensure the input image has the same size and resolution as our training data. To run locally, download pretrained weights (https://github.com/levente-1/GAMBAS/releases/tag/v1.0/latest_net_G.pth) and save them in your desired folder. Next, set --checkpoints_dir in base_options.py to the directory that contains the model's folder, not the model folder itself or the file within it (e.g. if the file is in '/path/to/your/checkpoints/t2_model/latest_net_G.pth', set this to '/path/to/your/checkpoints'). Finally, set --name in base_options.py to the model folder (this is where the script will look for 'latest_net_G.pth'), and run "test.py" as above.

### Docker

To run our Docker container, use the following command:

```bash
docker run --rm \
--gpus "device=0" \
-v "your/input/folder:/workspace/input/Testing_data" \
-v "your/output/folder:/workspace/output/Predictions" \
ghcr.io/levente-1/gambas:latest
```

Here is a command to run:

```bash
echo "This is a test code block."
```

This text comes after the code block.

Make sure your input folder contains all your input ultra-low-field images rigidly registered to our template, and saves as .nii or .nii.gz files.
