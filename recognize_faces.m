function YPred = recognize_faces(RGB)
% recognize_faces - map images of faces to people names    
    load('model');
    num_images = size(RGB,3);
    % Get grayscale images of the desired size
    G = arrayfun(@(j)imresize(im2gray(RGB{j},targetSize),...
                1:num_images);
    B = cat(3,Images{:});
    D = prod(targetSize);
    B = reshape(B,D,[]);

    % Normalizing data...';
    B = single(B)./256;

    % Extract features
    X = V' * B;
    % Predict faces
    YPred = predict(Mdl, X)
end


