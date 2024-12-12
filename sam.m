%----------------------------------------------------------------
% File:     sam.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Wed Dec 11 15:30:28 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------
% Image segmentation. This script demonstrates the use of a pretrained
% model to divide an image into "objects". One of the objects will be
% the person's face. Thus, we can focus on facial recognition rather than
% just picking the correct image out of a collection of, say, 2500 images
% The mask is precise enough to identify hair.

location = fullfile('lfw','Angelina_Jolie');
imds = imageDatastore(location);
while hasdata(imds)
    I = read(imds);
    [masks,scores] = imsegsam(I, MinObjectArea=1000, ...
                              ScoreThreshold=0.8,...
                              ExecutionEnvironment="auto",...
                              Verbose=true);
    labelMatrix = labelmatrix(masks);
    maskOverlay = labeloverlay(I,labelMatrix);
    imshow(maskOverlay,[])
    pause(5);
end
