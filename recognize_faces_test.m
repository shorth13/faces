location = fullfile('lfw');

disp('Creating image datastore...');
% We read images in their original format
imds = imageDatastore(location,...
                      'IncludeSubfolders',true,...
                      'LabelSource','foldernames');

load('model',["persons"])
