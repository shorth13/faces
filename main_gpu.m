targetSize=[128,128];
location = fullfile('lfw');
svd_cache = fullfile('cache','svd.mat');

disp('Creating image datastore...');
imds = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));
montage(preview(imds));
disp('Reading all images');
A = readall(imds);

B = cat(3,A{:});
imshow(B(:,:,1))
D = prod(targetSize);
B = reshape(B,D,[]);

disp('Normalizing data...');
B = single(B)./256;
%[N,C,SD] = normalize(B);
N=gpuArray(B);
if existsOnGPU(N)
    disp('Successfully moved image data array to GPU')
end

disp('Finding SVD...');
tic;
[Ugpu,Sgpu,Vgpu] = svd(N);
toc;

disp('Status of arrays:')
if existsOnGPU(Ugpu)
    disp('U is on GPU.')
end
if existsOnGPU(Vgpu)
    disp('V is on GPU.')
end
if existsOnGPU(Sgpu)
    disp('S is on GPU.')
end
U = gather(Ugpu);
V = gather(Vgpu);
S = gather(S);

plot(log(diag(S)),'.');
