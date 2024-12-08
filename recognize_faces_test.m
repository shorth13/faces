location = fullfile('lfw');

disp('Creating image datastore...');
% We read images in their original format
imds0 = imageDatastore(location,...
                       'IncludeSubfolders',true,...
                       'LabelSource','foldernames');

load('model',["persons"])

idx = ismember(imds0.Labels, persons);
imds = subset(imds0, idx);

RGB = readall(imds);

YPred = recognize_faces(RGB(1)');
