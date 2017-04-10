function p = predictOneVsAll(all_theta, X)
  m = size(X, 1);
  num_labels = size(all_theta, 1);
  X = [ones(m, 1) X];

  regressions = sigmoid(X * all_theta');
  [predictions, label_indices] = max(regressions, [], 2);
  p = label_indices;
end
