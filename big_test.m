location = fullfile('lfw');

disp('Creating image datastore...');
% We read images in their original format
imds0 = imageDatastore(location,...
                       'IncludeSubfolders',true,...
                       'LabelSource','foldernames');

load('big_model',["persons"])

idx = ismember(imds0.Labels, persons);
imds = subset(imds0, idx);

RGB = readall(imds);

Y = imds.Labels;

YPred = i_recognize_faces(RGB);

% Burn labels into the images
for j=1:numel(RGB)
    RGBannotated{j} = insertObjectAnnotation(RGB{j}, 'rectangle', [10,10,100,20], YPred(j));
end

montage(RGBannotated);

Accuracy = numel(find(Y==YPred))/numel(Y);
disp(['Percentage of correctly labeled images: ', num2str(100*Accuracy),'%']);


function YPred = i_recognize_faces(RGB)
% RECOGNIZE_FACES - map images of faces to people names    
%  YPred = recognize_faces(RGB) accepts a cell array RGB of images, which
% should be RGB images. YPred returns a categorical array of image
% labels.
    ;
    % Load precomputed model from a MAT file. For a format
    % of the file, see the M-file main_fitcecoc.m. 
    load('big_model.mat');
    num_images = size(RGB,3);
    % Get grayscale images of the desired size
    Grayscale = cellfun(@(I)imresize(im2gray(I),targetSize),...
                        RGB, 'uni',false);
    B = cat(3,Grayscale{:});
    D = prod(targetSize);
    B = reshape(B,D,[]);

    % Normalizing data...';
    B = single(B)./256;
    B = (B - C) ./ SD;

    % Extract features
    X = (U' * B) ./ S;
    % Predict faces
    YPred = predict(Mdl, X);
end


