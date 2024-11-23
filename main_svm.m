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
imds = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));
montage(preview(imds));
disp('Reading all images');
A = readall(imds);

% Play faces
if false
    for j=1:length(A)
        imshow(A{j}),title(imds.Labels(j),'Interpreter','none');
        colormap gray;
        drawnow;
        pause(1);
    end
end

B = cat(3,A{:});
imshow(B(:,:,1))
D = prod(targetSize);
B = reshape(B,D,[]);

disp('Normalizing data...');
B = single(B)./256;
%[N,C,SD] = normalize(B);
N=B;

if exist(svd_cache,'file') == 2
    disp('Loading SVD from cache...');
    load(svd_cache)
else
    disp('Finding SVD...');
    tic;
    [U,S,V] = svd(N,'econ');
    toc;
    disp('Writing SVD cache...')
    save(svd_cache,'U','S','V');
end

k=256;

disp('Training Support Vector Machine...');
X0 = V(:,1:k)';
person1 = 'Angelina_Jolie';
person2 = 'George_W_Bush';

idx1 = find(imds.Labels==person1);
idx2 = find(imds.Labels==person2);

X = horzcat(X0(:,idx1),X0(:,idx2));

Y = vertcat(imds.Labels(idx1),imds.Labels(idx2));

Mdl = fitcsvm( X, Y,'Verbose',true);


W = X(3949,:);
I = reshape(U(:,1:k)*S(1:k,1:k)*W',targetSize);
imagesc(I);
colormap gray;
drawnow;

disp('Running prediction...');
[label, score] = predict(Mdl, W)
