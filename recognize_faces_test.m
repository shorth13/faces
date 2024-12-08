location = fullfile('lfw');

disp('Creating image datastore...');
imds = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames');

load('model',["persons"])
