targetSize = [128,128];
k=120;                                   % Number of features to consider
location = fullfile('lfw');

disp('Creating image datastore...');
imds0 = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));

disp('Creating subset of several persons...');
tbl = countEachLabel(imds0);
mask = tbl{:,2}>=10 & tbl{:,2}<=20;
disp(['Number of images: ',num2str(sum(tbl{mask,2}))]);

persons = unique(tbl{mask,1});

% Limit to people being recognized
[lia,locb] = ismember(imds0.Labels, persons);

imds = subset(imds0, lia);

t=tiledlayout('flow');
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

% Get an montage of eigenfaces
Eigenfaces = arrayfun(@(j)reshape((U(:,j)-min(U(:,j)))./(max(U(:,j))-min(U(:,j))),targetSize), ...
    1:size(U,2),'uni',false);

nexttile(t);
montage(Eigenfaces(1:16));
title('Top 16 Eigenfaces');
colormap(gray);

% NOTE: Rows of V are observations, columns are features.
% Observations need to be in rows.
k = min(size(V,2),k);

% Discard unnecessary data
V = V(:,1:k);
S = diag(S);
S = S(1:k);
U = U(:,1:k);

% Find feature vectors of all images
X0 = V;


X = X0;
Y = imds.Labels;

% Create colormap
cm=jet;
% Assign colors to target values
c=cm(1+ mod(uint8(Y),size(cm,1)-1),:);

disp('Training Support Vector Machine...');
options = statset('UseParallel',true);
tic;
Mdl = fitcecoc(X, Y,'Verbose', 0,'Learners','svm','Options',options);
toc;

%[YPred,Score] = predict(Mdl,X);
[YPred,Score,Cost] = resubPredict(Mdl);

disp(['Fraction of correctly predicted images:', ...
      num2str(numel(find(YPred==Y))/numel(Y))]);

% Save the model and persons that the model recognizes.
% NOTE: An important part of the submission.
save('model','Mdl','persons','U','targetSize');
