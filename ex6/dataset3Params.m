function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

test_params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
num_of_params = length(test_params);
minimum_error = realmax("double");

for c_index = 1:num_of_params
  c_value = test_params(c_index);

  for s_index = 1:num_of_params
    sigma_value = test_params(s_index);
    model = svmTrain(X, y, c_value, @(x1, x2) gaussianKernel(x1, x2, sigma_value));
    predictions = svmPredict(model, Xval);
    error_val = mean(double(predictions ~= yval));
    if (error_val < minimum_error)
      minimum_error = error_val;
      C = c_value;
      sigma = sigma_value;
    %elseif (error_val == minimum_error)
    %  fprintf('error_val == minimum_error...........................\n');
    endif
  endfor
endfor


% =========================================================================

end
