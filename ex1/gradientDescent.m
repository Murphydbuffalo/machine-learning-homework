function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  % number of training examples (rows)
  m = length(y);
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters
      H = X * theta;
      differences = H - y;
      derivatives = [mean(differences.* X(:, 1)); mean(differences.* X(:, 2))];
      theta = theta - (alpha * derivatives);

      % Save the cost J in every iteration
      J_history(iter) = computeCost(X, y, theta);
  end
end
