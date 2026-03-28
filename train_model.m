%% Preprocessing - cropping pipeline
% How to run this section:
% 1. Run section -> Select your raw Healthy folder -> Select an empty Healthy_Cropped folder.
% 2. Run section again -> Select your raw GON+ folder -> Select an empty GON_Cropped folder.

% 1. Ask the user to select the folder with the RAW images
disp('Select the folder containing your RAW images...');
inputFolder = uigetdir('', 'Select Input Folder (Raw Images)');
if inputFolder == 0
    disp('Canceled');
    return;
end

% 2. Ask the user where to save the CROPPED images
disp('Select the folder to save the CROPPED images...');
outputFolder = uigetdir('', 'Select Output Folder (Cropped Images)');
if outputFolder == 0
    disp('Canceled by user.');
    return;
end

% 3. Find all images in the chosen input folder
imageFiles = dir(fullfile(inputFolder, '*.*'));
% Filter out hidden system files (like . or ..)
imageFiles = imageFiles(~ismember({imageFiles.name}, {'.', '..'}));

% 4. Set up a progress bar
totalImages = length(imageFiles);
h = waitbar(0, 'Initializing dataset crop...');

% 5. Loop through every single image
for i = 1:totalImages
    try
        % Update progress bar
        waitbar(i/totalImages, h, sprintf('Processing Image %d of %d...', i, totalImages));
        
        % Load current image
        baseFileName = imageFiles(i).name;
        fullFileName = fullfile(inputFolder, baseFileName);
        img = imread(fullFileName);
        [rows, cols, ~] = size(img);
        
        % --- THE OPTIC cup TRACKER ---
        grayImg = rgb2gray(img);
        eyeMask = grayImg > 20; % Mask to ignore pure black
        
        blurredImg = imgaussfilt(grayImg, 30); % Heavy blur
        blurredImg(~eyeMask) = 0; % Force black borders to zero
        
        % Find the glowing nerve cup
        [~, maxIndex] = max(blurredImg(:));
        [centerY, centerX] = ind2sub(size(blurredImg), maxIndex);
        
        % Define crop box (40% of original width)
        boxSize = round(cols * 0.40); 
        halfBox = round(boxSize / 2);
        
        top = max(1, centerY - halfBox);
        bottom = min(rows, centerY + halfBox);
        left = max(1, centerX - halfBox);
        right = min(cols, centerX + halfBox);
        
        % Slice the image matrix
        imgCropped = img(top:bottom, left:right, :);
        % -------------------------------
        
        % Standardize size for ResNet-50
        imgResized = imresize(imgCropped, [224 224]);
        
        % Save it to the new output folder
        outputFileName = fullfile(outputFolder, baseFileName);
        imwrite(imgResized, outputFileName);
        
    catch
        % If an image is corrupted or not a picture, skip it without crashing
        disp(['️Error processing', baseFileName, ' - Skipping.']);
    end
end

% Clean up
close(h);
disp('Preprocessing done');

%% Training the model

% 1. Point MATLAB to your image folders
% Classifies images as 'Glaucoma' or 'Healthy' based on folder names
dataPath = 'G:\My Drive\HACKATHON\dataset_cropped';
imds = imageDatastore(dataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% 2. Split the data (80% for training, 20% for testing)
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% 3. Load ResNet-50 
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
disp('Training begin');
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

% 7. Save the trained model to use in our App later
save('glaucoma_model.mat', 'trainedNet');
disp('Training complete and model saved');


%% CONFUSION MATRIX
%%IMPORTANT
%Run this section first before proceeding to other section

% Load model and validation data
load('glaucoma_model.mat', 'trainedNet');

% Get the AI's predictions on validation data
[predictedLabels, ~] = classify(trainedNet, augimdsValidation);
actualLabels = imdsValidation.Labels;
[predictedLabels, scores] = classify(trainedNet, augimdsValidation);
actualLabels = imdsValidation.Labels;

% 2. Calculate raw accuracy
accuracy = sum(predictedLabels == actualLabels) / numel(actualLabels);

% Generate and show ONLY the Confusion Matrix
figure('Name', 'Clinical Confusion Matrix');
confusionchart(actualLabels, predictedLabels, ...
    'Title', 'Confusion Matrix');

%% ROC AUC
% --- 1. Identify the Positive Class ---
% The 'scores' variable has two columns of probabilities (Healthy and GON+).
% We need to grab specifically the column for our positive pathology (GON+).
classes = trainedNet.Layers(end).Classes;
positiveClassIdx = find(classes == 'GON+'); 

% Extract just the confidence percentages for GON+
gonProbabilities = scores(:, positiveClassIdx);

% --- 2. Calculate the ROC Curve Math ---
% 'perfcurve' is MATLAB's built-in performance curve calculator
[X, Y, T, AUC] = perfcurve(actualLabels, gonProbabilities, 'GON+');

% --- 3. Plot the Graph ---
figure('Name', 'Clinical ROC Curve', 'Position', [100, 100, 600, 500]);
plot(X, Y, 'b-', 'LineWidth', 3); % The AI's performance line
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1.5); % The "Random Guess" diagonal line

% Add professional labels and titles
xlabel('False Positive Rate (1 - Specificity)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('True Positive Rate (Sensitivity)', 'FontSize', 12, 'FontWeight', 'bold');
title('ROC Curve: Glaucoma Detection (ResNet-50)', 'FontSize', 14);
legend(['Model Performance (AUC = ', num2str(AUC, '%.4f'), ')'], 'Random Guess (AUC = 0.50)', 'Location', 'southeast');

grid on;
hold off;

disp(['AUC = ', num2str(AUC)]);

%% Calculate precision, recall, accuracy & F1 score
% --- Clinical Performance Metrics Calculator ---
% This script calculates Accuracy, Precision, Recall, and F1-Score
% assuming 'actualLabels' and 'predictedLabels' are in your workspace.

% 1. Define the Positive Pathological Class
% For this project, detecting Glaucoma (GON+) is our primary target.
positiveClass = categorical({'GON+'});

% 2. Extract Confusion Matrix Variables using Logical Arrays
% True Positives: Actual is GON+, Predicted is GON+
TP = sum((actualLabels == positiveClass) & (predictedLabels == positiveClass));

% True Negatives: Actual is Healthy, Predicted is Healthy
TN = sum((actualLabels ~= positiveClass) & (predictedLabels ~= positiveClass));

% False Positives: Actual is Healthy, Predicted is GON+ (False Alarm)
FP = sum((actualLabels ~= positiveClass) & (predictedLabels == positiveClass));

% False Negatives: Actual is GON+, Predicted is Healthy (Missed Diagnosis)
FN = sum((actualLabels == positiveClass) & (predictedLabels ~= positiveClass));

% 3. Calculate the Core Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN);
precision = TP / (TP + FP);
recall = TP / (TP + FN); % Clinically known as Sensitivity
f1_score = 2 * (precision * recall) / (precision + recall);

% 4. output
fprintf('\n=== Model Performance ===\n');
fprintf('Accuracy  : %.4f \n', accuracy);
fprintf('Precision : %.4f \n', precision);
fprintf('Recall    : %.4f \n', recall);
fprintf('F1-Score  : %.4f \n', f1_score);
fprintf('======================================\n\n');

%% Feature Extraction

% 1. Load your trained model
disp('Loading model...');
load('glaucoma_model.mat', 'trainedNet');

% 2. Open a file chooser so you can pick a specific image to analyze
% (Tip: Pick one from your dataset_cropped folder!)
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files'}, 'Select a cropped fundus image');
if isequal(file, 0)
    disp('Canceled');
    return; 
end

% Read and resize the image for ResNet
img = imread(fullfile(path, file));
imgResized = imresize(img, [224 224]);

disp('Extracting features from the neural network...');

% 3. Look at an EARLY layer 
earlyLayer = 'conv1';
earlyFeatures = activations(trainedNet, imgResized, earlyLayer);

% 4. Look at a DEEP layer 
deepLayer = 'activation_40_relu';
deepFeatures = activations(trainedNet, imgResized, deepLayer);

% 5. Plot them side-by-side
figure('Name', 'Feature Extraction Comparison', 'Position', [100, 100, 1200, 400]);

% Panel 1: Original Image
subplot(1,3,1); 
imshow(imgResized); 
title('1. Original Input');

% Panel 2: Early Layer
subplot(1,3,2); 
% We grab the first 16 feature maps and tile them into a grid
imshow(imtile(mat2gray(earlyFeatures(:,:,1:16)))); 
title('2. Early Features (conv1\_relu)');

% Panel 3: Deep Layer
subplot(1,3,3); 
imshow(imtile(mat2gray(deepFeatures(:,:,1:16)))); 
title('3. Deep Features (activation\_40\_relu)');

disp('Feature extraction ready');
