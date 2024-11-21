location = fullfile('lfw');
imds = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames');
