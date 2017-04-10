function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  n = size(X, 2);
  m = length(X);
  mu = zeros(1, n);
  sigma = zeros(1, n);
  fprintf('n is %f, m is %f', n, m);

  for columnIndex = 1:n
    feature = X(:, columnIndex);
    mu(1, columnIndex) = mean(feature);
    sigma(1, columnIndex) = std(feature)

    for rowIndex = 1:m
      X_norm(rowIndex, columnIndex) = (X(rowIndex, columnIndex) - mu(1, columnIndex)) / sigma(1, columnIndex);
    end
  end
end
