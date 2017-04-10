function [J, grad] = lrCostFunction(theta, X, Y, lambda)
  % number of training examples
  m = length(Y);

  % Predicted values given parameters theta and training data X passed to the logistic function.
  H = sigmoid(X * theta);

  % Cost function - how much are the predicted values, H, off by on average
  % compared to the known results, Y?
  meanError = (-1 * ( (Y' * log(H)) + ((1 - Y)' * log(1 - H)) ) / m);

  % Regularization - reduce all theta values, except the bias unit theta1, to
  % prevent overfitting. We do this by increasing the cost of each theta by some
  % large amount, lambda * theta^2 in this case.
  regularizationAmount = (lambda / (2 * m)) * sum(theta(2:end) .^ 2);
  J = meanError + regularizationAmount;

  derivatives = (X' * (H - Y)) / m;
  regularizationDerivatives = (lambda / m) * theta;
  regularizationDerivatives(1) = 0;
  grad = derivatives + regularizationDerivatives;
end
