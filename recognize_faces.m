function YPred = recognize_faces(F)
% recognize_faces - map images of faces to people names    
    load('model');
    % Extract features
    X = V' * F;
    % Predict faces
    YPred = predict(Mdl, X)
end

