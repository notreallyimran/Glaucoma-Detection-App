# Glaucoma Detection App using ResNet-50 Neural Network and Grad-CAM XAI
# Project Overview
This Matlab-powered application classifies DFIs into glaucoma positive (GON+). In addition, this model mitigates Clever Hans effect by applying cropping pipelining which enable the model to focus on optic cup instead of random spots on DFIs.

Our ResNet-50 Neural Network is trained with DFIs from Hillel Yaffe Glaucoma Dataset - https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/


# Project Files
1. train_model.m - Contains all the working code
2. app1.mlapp - Runs our detection app
3. projectworkspace.mat [ https://drive.google.com/file/d/19TZBsUBiqwZvrtBxaPBgHurL9zbI6muk/view?usp=sharing ] - contains saved workspace, trained model

# Before Running the App
1. This app only works on Matlab R2024a and later version
2. Download projectworkspace.mat 
3. Make sure all files are in the same directory
4. Install these Toolboxes on Matlab:
   - deep learning toolbox
   - deep learning toolbox for resnet-50 network
   - image processing toolbox

# Using the App
1. Open app1.mlapp, this may take a while
2. Upload any DFI from dataset
3. Wait for model's verdict

# FAQ
1. I can't run the app
    - Install the toolboxes
    - Ensure app1.mlapp and glaucoma_model.mat are in the same directory
2. I 've installed the toolboxes and it's still not running. Please Follow these steps:
      - Follow the instructions in train_model.m to run the script
      - type appdesigner on command window>load app1.mlapp
# Authors
1. Imran Fareez Azmy - imranfareez1@gmail.com
2. Ahmad Nafis Mohd Zulkiifli - ahmadnafiszulkifli@gmail.com
