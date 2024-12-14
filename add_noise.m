%----------------------------------------------------------------
% File:     add_noise.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Sat Dec 14 12:34:12 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------
% Example: Adding white noise to a 128x128 grayscale image

% Load or create a grayscale image with values in the range [0, 1]
image = imread('peppers.png'); % Replace this with your actual image

% Specify the noise level (variance of the noise)
noiseLevel = 0.01; % Adjust this to control the noise intensity

% Add white Gaussian noise
noisyImage = imnoise(image, 'gaussian', 0, noiseLevel);

% Display the original and noisy images
figure;
subplot(1, 2, 1);
imshow(image, []);
title('Original Image');

subplot(1, 2, 2);
imshow(noisyImage, []);
title('Noisy Image');
