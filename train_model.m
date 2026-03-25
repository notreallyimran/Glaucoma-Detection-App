% 1. Point MATLAB to your image folders
% This automatically labels images as 'Glaucoma' or 'Healthy' based on folder names
dataPath = 'G:\My Drive\HACKATHON\dataset_cropped';
imds = imageDatastore(dataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% 2. Split the data (80% for training, 20% for testing)
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% 3. Load ResNet-50 and adapt it for our images
net = resnet50;
inputSize = net.Layers(1).InputSize; % ResNet needs 224x224 images

% Automatically resize our images to fit ResNet
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

% 4. Modify the network for Binary Classification (2 classes instead of 1000)
lgraph = layerGraph(net);
newFc = fullyConnectedLayer(2, 'Name', 'new_fc', 'WeightLearnRateFactor', 10);
newClass = classificationLayer('Name', 'new_classoutput');

lgraph = replaceLayer(lgraph, 'fc1000', newFc);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClass);

% 5. Set Training Rules
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 5, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 6. Train the Network! (This will take some time)
disp('Starting training...');
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

% 7. Save the trained model to use in our App later
save('glaucoma_model.mat', 'trainedNet');
disp('Training complete and model saved!');