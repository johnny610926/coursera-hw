function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_y_diff = (X * theta) - y;
theta_ones = ones(size(theta)); theta_ones(1) = 0; % No need to regularize theta(1) (which corresponds to theta_0)

J = (1/(2*m)) * ((h_y_diff' * h_y_diff) + lambda * (theta'* (theta.*theta_ones)));
grad = (1/m) * (h_y_diff'*X)' + ((lambda/m)*(theta.*theta_ones));


% =========================================================================

grad = grad(:);

end
