function YPred = recognize_faces(Images)
% recognize_faces - map images of faces to people names    
    load('model');
    B = cat(3,Images{:});
    D = prod(targetSize);
    B = reshape(B,D,[]);

    disp('Normalizing data...');
    B = single(B)./256;

    % Extract features
    X = V' * F;
    % Predict faces
    YPred = predict(Mdl, X)
end


