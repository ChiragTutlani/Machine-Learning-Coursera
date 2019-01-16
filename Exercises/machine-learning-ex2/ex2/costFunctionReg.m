function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Computing Cost Function
logHypothesis = log(sigmoid(X*theta));
log1minusHypothesis =  log(1-sigmoid(X*theta));
A = ((-y'*logHypothesis)-((1-y)'*log1minusHypothesis))/m;
B = 0;
for i=2:length(theta)
  B = B + theta(i,1)*theta(i,1);
endfor
B = (lambda*B)/(2*m);
J = A+B;

%Computing Gradient
% Using regularization for every theta
grad = ((X'*(sigmoid(X*theta)-y))+lambda*theta)/m;
% Changing theta_0 to unregularized
grad(1,1) = (X(:,1)'*(sigmoid(X*theta)-y))/m;


% =============================================================

end
