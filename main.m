targetSize=[128,128];
location = fullfile('lfw');

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
    load(svd_cache)
else
    disp('Finding SVD...');
    tic;
    [U,S,V] = svd(N,'econ');
    toc;
    save(svd_cache,'U','S','V');
end

if false
    for j=1:size(U,2)
        imagesc(reshape(U(:,j), targetSize));
        title([num2str(j),': ',num2str(S(j,j))]);
        pause(1);
    end
end

% N = U*S*V';
k=512;Z=U(:,1:k)*S(1:k,1:k)*V(:,1:k)';

colormap gray;
if false
    for j=1:size(Z,2)
        imagesc(reshape(Z(:,j),targetSize));
        title(imds.Labels(j),'Interpreter','none');
        pause(0.5);
    end

end

disp('Training Support Vector Machine...');
X = V(:,1:k);
Y = imds.Labels=='George_W_Bush';
% Map to +1/-1
Y = 2.*Y-1;

tTree = templateTree('surrogate','on');
tEnsemble = templateEnsemble('GentleBoost',100,tTree);

%mdl = fitcsvm( X, Y,'Verbose',true);
options = statset('UseParallel',true,'Verbose',2);
Mdl = fitcecoc(X,Y,'Coding','onevsall','Learners',tEnsemble,...
                'Prior','uniform','NumBins',50,'Options',options);

disp('Testing on "Dabya"...');
W = X(3949,:);
I = reshape(U(:,1:k)*S(1:k,1:k)*W',targetSize);
imagesc(I);
[label, score] = predict(Mdl, W)
