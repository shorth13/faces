targetSize = [128,128];
k=60;                                   % Number of features to consider
location = fullfile('lfw');

disp('Creating image datastore...');
imds0 = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));

disp('Creating subset of several persons...');
tbl = countEachLabel(imds0);
idx = find(tbl{:,2}>=10);
disp(['Number of images: ',num2str(sum(tbl{idx,2}))]);

persons = unique(tbl{idx,1});

imds = subset(imds0, idx);

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

% Limit to people being recognized
[lia,locb] = ismember(imds.Labels, persons);

X = X0(lia,:);
Y = imds.Labels(lia);
cats = persons;
Y=categorical(Y,cats);

% Create colormap
cm=jet;
% Assign colors to target values
c=cm(1+ mod(uint8(Y),size(cm,1)-1),:);

disp('Training Support Vector Machine...');
options = statset('UseParallel',true);
tic;
Mdl = fitcecoc(X, Y,'Verbose', 2,'Learners','svm','Options',options);
toc;

% Generate a plot in feature space using top two features
nexttile(t);
scatter3(X(:,1),X(:,2),X(:,3),50,c);
title('A top 3-predictor plot');
xlabel('x1');
ylabel('x2');
zlabel('x3');

nexttile(t);
scatter3(X(:,4),X(:,5),X(:,6),50,c);
title('A next 3-predictor plot');
xlabel('x4');
ylabel('x5');
zlabel('x6');

%[YPred,Score] = predict(Mdl,X);
[YPred,Score,Cost] = resubPredict(Mdl);

disp(['Fraction of correctly predicted images:', ...
      num2str(numel(find(YPred==Y))/numel(Y))]);

% Save the model and persons that the model recognizes.
% NOTE: An important part of the submission.
save('model','Mdl','persons','U','targetSize');
