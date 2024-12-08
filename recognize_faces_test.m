location = fullfile('lfw');

disp('Creating image datastore...');
% We read images in their original format
imds0 = imageDatastore(location,...
                       'IncludeSubfolders',true,...
                       'LabelSource','foldernames');

load('model',["persons"])

RGB = readimage(imds,1);
