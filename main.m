targetSize=[128,128];
location = fullfile('lfw');

imds = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames');
[w,h,c]=size(imds.readimage(1));
ds = transform(imds,@(I)imresize(im2gray(I),targetSize),'IncludeInfo',false);
montage(preview(ds));
