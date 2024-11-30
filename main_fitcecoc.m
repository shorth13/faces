%----------------------------------------------------------------
% File:     main_svm.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Fri Nov 22 20:02:05 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------
% Binary classification
% Distinguish between two persons (Angenlina Jolie and Eduardo Duhalde).
targetSize=[128,128];
k=25;                                   % Number of features to consider

t=tiledlayout('flow');

disp('Creating image datastore...');
imds0 = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));

disp('Creating subset of 2 persons...');
persons = {'Angelina_Jolie', 'Eduardo_Duhalde', 'George_W_Bush'}

idx = num2cell(imds0.Labels==persons,2);



idx0 = find(mask0);

imds = subset(imds0, idx0);
nexttile(t);
montage(imds);

disp('Reading all images');
A = readall(imds);

B = cat(3,A{:});
D = prod(targetSize);
B = reshape(B,D,[]);

disp('Normalizing data...');
B = single(B)./256;
[B,C,SD] = normalize(B);
tic;
[U,S,V] = svd(B,'econ');
toc;


% NOTE: Rows of V are observations, columns are features.
% Observations need to be in rows.
X0 = V(:,1:k);

mask1 = imds.Labels==person1;
mask2 = imds.Labels==person2;
mask = mask1|mask2;
idx = find(mask);

X = X0(idx,:);
Y = imds.Labels(idx);

% Limit the number of categories to just two
cats = {person1,person2};
Y=categorical(Y,cats);

% Create colormap
cm=[1,0,0;
    0,0,1];
% Assign colors to target values
c=cm(uint8(Y),:);

disp('Training Support Vector Machine...');
tic;
Mdl = fitcsvm(X, Y,'Verbose', 1);
toc;

% ROC = receiver operating characteristic
% See https://en.wikipedia.org/wiki/Receiver_operating_characteristic
disp('Plotting ROC metrics...');
cv = crossval(Mdl);
rm = rocmetrics(cv);
nexttile(t);
plot(rm);

% Generate a plot in feature space using top two features
nexttile(t);
scatter(X(:,1),X(:,2),60,c);
title('A 2-predictor plot');
xlabel(cv.PredictorNames(1));
ylabel(cv.PredictorNames(2));

%[YPred,Score] = predict(Mdl,X);
[YPred,Score,Cost] = resubPredict(Mdl);

disp('Plotting confusion matrix...')
nexttile(t);
confusionchart(Y, YPred);
title(['Number of features: ' ,num2str(k)]);

% Get an montage of eigenfaces
Eigenfaces = arrayfun(@(j)reshape((U(:,j)-min(U(:,j)))./(max(U(:,j))-min(U(:,j))),targetSize), ...
    1:size(U,2),'uni',false);

%nexttile(t);
figure;
montage(Eigenfaces(1:16));
colormap(gray);
