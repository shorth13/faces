%----------------------------------------------------------------
% File:     main.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Fri Nov 22 20:02:05 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------
% A basic face recognition system workflow
%
targetSize=[128,128];
location = fullfile('lfw');
svd_cache = fullfile('cache','svd.mat');

disp('Creating image datastore...');
imds0 = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));
person1 = 'Angelina_Jolie';
person2 = 'Eduardo_Duhalde';

mask0_1 = imds0.Labels==person1;
mask0_2 = imds0.Labels==person2;
mask0  = mask0_1|mask0_2;
idx = find(mask0);

imds = subset(imds0, idx);
montage(preview(imds));

disp('Reading all images');
A = readall(subds);

B = cat(3,A{:});
imshow(B(:,:,1))
D = prod(targetSize);
B = reshape(B,D,[]);

disp('Normalizing data...');
B = single(B)./256;
[B,C,SD] = normalize(B);
tic;
[U,S,V] = svd(B,'econ');
toc;

k=3;

disp('Training Support Vector Machine...');
% NOTE: Rows of V are observations, columns are features.
% Observations need to be in rows.
X0 = V(:,1:k);

mask1 = imds.Labels==person1;
mask2 = imds.Labels==person2;
mask = mask1|mask2


X = X0(mask,:);

L = imds.Labels(mask);
Y = L==person1;

cm=[1,0,0;
    0,0,1];
c=cm(1+Y,:);
scatter3(X(:,1),X(:,2),X(:,3),60,c);


Mdl = fitcsvm(X, Y,'Verbose', 2);

cv = crossval(Mdl);

[label,Score,Cost] = resubPredict(Mdl);


