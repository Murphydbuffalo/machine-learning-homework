function J = computeCost(X, y, theta)
  % number of training examples (rows)
  m = length(y);

  H = X * theta;
  squaredDifference = (H - y).^2;
  J = sum(squaredDifference) / (2 * m);
  fprintf('Mean squared error, J(0), is %f...\n', J)
end
