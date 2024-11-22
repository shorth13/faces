targetSize=[128,128];
location = fullfile('lfw');

disp('Creating image datastore...');
imds = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));
%montage(preview(imds));
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
[N,C,SD] = normalize(B);

disp('Finding SVD...');
tic;
[U,S,V] = svd(N,'econ');
toc;

if false
    for j=1:size(U,2)
        imagesc(reshape(U(:,j), targetSize));
        title([num2str(j),': ',num2str(S(j,j))]);
        pause(1);
    end
end

% N = U*S*V';
k=512;Z=U(:,1:k)*S(1:k,1:k)*V(:,1:k)';

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
mdl = fitcecoc(X, Y,'Coding','onevsall','Learners','svm','Verbose',true);

disp('Testing on "Dabya"...');
W = X(3949,:);
I = reshape(U(:,1:k)*S(1:k,1:k)*W',targetSize);
imagesc(I);
[label, score] = predict(mdl, W)
