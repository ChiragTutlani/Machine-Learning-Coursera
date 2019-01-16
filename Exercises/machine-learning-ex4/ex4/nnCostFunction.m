function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1), X];

% ith column of a2 contains activation value of ith example  
a2 = sigmoid(Theta1*X');
a2 = [ones(1,m) ; a2];
% a2 : 26 X 5000

% a3 : 10 X 5000
% Every column of a3 contains prediction on each example
a3 = sigmoid(Theta2*a2);
  

% Making y according to 0/1 
temp = zeros(m,num_labels);
for i=1:m
  temp(i,y(i)) = 1;
endfor
y = temp;

% Calculate J (unregularized) for every output class
A = zeros(num_labels,1);
for i=1:num_labels
  logHypothesis = log(a3(i,:));
  log1minusHypothesis =  log(1-a3(i,:));
  A(i,1) = (((-logHypothesis*y(:,i))-(log1minusHypothesis*(1-y(:,i))))/m);  
endfor
J = sum(A);

% Calculating the regularized part of cost function

temp1 = Theta1.*Theta1;
temp2 = Theta2.*Theta2;
temp1(:,1)=0;
temp2(:,1)=0;

B = (sum(temp1(:))+sum(temp2(:)))*(lambda/(2*m));
J=J+B;


% -------------------------------------------------------------
% Computing gradient
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for t=1:m
  delta3 = a3(:,t) - y(t,:)';
  delta2 = Theta2'*delta3.*sigmoidGradient([1;Theta1*X(t,:)']);
  Delta1 = Delta1 + (delta2(2:end)*[X(t,:)]);
  Delta2 = Delta2 + (delta3*a2(:,t)');
endfor
% temp theta for regularization
temp_theta = Theta1;
temp_theta(:,1)=0;
Theta1_grad = Delta1/m + (lambda/m)*temp_theta;
temp_theta = Theta2;
temp_theta(:,1)=0;
Theta2_grad = Delta2/m + (lambda/m)*temp_theta;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
