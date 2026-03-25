% === AUTOMATED OPTIC NERVE DATASET GENERATOR ===

% 1. Ask the user to select the folder with the RAW images
disp('Select the folder containing your RAW images...');
inputFolder = uigetdir('', 'Select Input Folder (Raw Images)');
if inputFolder == 0
    disp('Canceled by user.');
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
        
        % --- THE OPTIC NERVE TRACKER ---
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
        disp(['⚠️ Error processing: ', baseFileName, ' - Skipping.']);
    end
end

% Clean up
close(h);
disp('✅ Dataset generation complete! All images are now centered on the Optic Nerve.');