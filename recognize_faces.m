function YPred = recognize_faces(RGB)
% recognize_faces - map images of faces to people names    
    load('model');
    num_images = size(RGB,3);
    % Get grayscale images of the desired size
    Grayscale = cellfun(@(I)imresize(im2gray(I),targetSize),...
                        RGB);
    B = cat(3,Grayscale{:});
    D = prod(targetSize);
    B = reshape(B,D,[]);

    % Normalizing data...';
    B = single(B)./256;

    % Extract features
    X = V' * B;
    % Predict faces
    YPred = predict(Mdl, X)
end


