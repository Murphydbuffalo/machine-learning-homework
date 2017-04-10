function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  % number of features (columns)
  n = length(theta);

  % number of training examples (rows)
  m = length(y);

  derivatives = zeros(n, 1);
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters
      H = X * theta;
      differences = H - y;

      for i = 1:n
        derivatives(i, 1) = mean(differences.* X(:, i));
      end

      theta = theta - (alpha * derivatives);

      % Save the cost J in every iteration
      J_history(iter) = computeCostMulti(X, y, theta);
  end
end
