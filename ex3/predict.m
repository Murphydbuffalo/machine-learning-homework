function p = predict(Theta1, Theta2, X)
  m = size(X, 1);
  X = [ones(m, 1), X];
  num_labels = size(Theta2, 1);
  % Each unit (in any layer) is a logistic regression expecting argument z = theta' * X
  % Layer one should have 25 units
  % Layer two should have 10 units (one for each class/digit)
  % X = 5000 x 401 ...in this exercise we have already trained the models and found
  % the optimal values for theta. So, for this function we are using X as data
  % to make predictions with, and NOT as training data to use w/ gradient descent
  % or some similar optimization algorithm.

  % Theta1 = 25 x 401 ... features for each pixel, plus the constant bias unit of 1
  % Theta2 = 10 x 26 ...features for the second layer are the outputs from the first layer
  % z2 = Theta1(25 x 401) x X'(401 x 5000) => 25 x 5000
  % result = Theta2(10 x 26) x z2(26 x 5000) => 10 x 26
  % [prediction, indexes] = max(result)
  layer_one_output = sigmoid(Theta1 * X');
  z2 = [ones(1, size(layer_one_output, 2)); layer_one_output];
  disp(size(Theta2))
  disp(size(z2))
  layer_two_output = sigmoid(Theta2 * z2);
  [output_layer_predictions, indices] = max(layer_two_output);
  p = indices;
end
