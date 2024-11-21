targetSize=[128,128];
location = fullfile('lfw');

imds = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));
%montage(preview(imds));
A=readall(imds);

% Play faces
for j=1:length(A)
    imshow(A{j}),title(imds.Labels(j),'Interpreter','none');
    pause(1);
end
