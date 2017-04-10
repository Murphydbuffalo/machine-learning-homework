function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  % `X` is an `m` x `n + 1` matrix containing all training data.
  % There are 5000 example handwritten digits, so m = 5000
  % Each handwritten example is a 20 pixel by 20 pixel grayscale image.
  % "Flattening" out that 20 x 20 image into a 400 pixel long vector gives us
  % n = 400 features or columns (each pixel is a feature).

  % y is an `m` x `1` column vector showing the actual digit each row in the
  % training set `X` corresponds to. 0 is given as 10 in the matrix y for ease of
  % setting the index `c` in the for loop below.

  % num_labels is the number of classes for our multiclass classification problem
  % in this case there are 10 classes: digits 1 through 9 plus 0.

  % The job of this function is to return a `num_labels` x `n + 1` matrix containing
  % the optimal parameters/weights theta for our logistic regression functions.
  % To find those ideal parameters we will pass the regularized cost function and
  % partial derivatives to `fmincg` which is a more efficient alternative to the
  % gradient descent algorithm for finding the values of theta which minimize the
  % the cost function J(theta).

  m = size(X, 1);
  n = size(X, 2);
  % Add constant 1 to each row in X for column 1 (the bias unit)
  X = [ones(m, 1), X];

  all_theta = zeros(num_labels, n + 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);

  for c = 1:num_labels
    % Is the corresponding row in `X` in the current class, `c`?
    isCurrentClass = y == c;
    initial_theta = zeros(n + 1, 1);
    % Run fmincg to obtain the optimal theta
    % This function will return theta and the cost
    [theta] = ...
        fmincg (@(t)(lrCostFunction(t, X, isCurrentClass, lambda)), ...
                initial_theta, options);

    all_theta(c, :) = theta(:);
  end
end
