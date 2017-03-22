function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    theta = theta - alpha * (1/m) * (((X * theta) - y)' * X)';
    % theta = theta - alpha * (1/m) * (X' * ((X * theta) - y)); % This seems be a bit more efficient than the above statement, because less one transpose

    J = computeCost(X, y, theta);

    if iter > 1,
      J_delta = J - J_history(iter - 1);
      % disp(sprintf("J(%d) = %0.4f and gradient = %0.4f", iter, J, J_delta));
      if J_delta >= 0,
        break;
      end;
    end;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = J; % J_history(iter) = computeCost(X, y, theta);

end

end
