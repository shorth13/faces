targetSize=[128,128];
location = fullfile('lfw');

imds = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(I)imresize(im2gray(I),targetSize));
montage(preview(imds));

A=readall(imds);
