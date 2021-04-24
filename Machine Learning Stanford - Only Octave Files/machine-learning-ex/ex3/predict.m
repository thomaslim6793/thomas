function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Input layer
A1 = [ones(m, 1) X] % X is m*n, so A_1 is m*(n+1)

% Second layer, bias term added
Z2 = A1 * Theta1'
A2 = [ones(m,1) sigmoid(Z2)] % A_1*Theta1' is 
% m*(n+1) * (n+1)*k = m*k. Adding column of 1's, m*(k+1). k is number of a

% Final layer, no bias term.
Z3 = A2 * Theta2'
A3 = sigmoid(Z3) % m*(k+1) * (k+1)*c = m*c. c is number of classes

% A_3 is the final layer so every element of A_3 is a prediction. different
% row denotes a different sample object, and different column denotes different 
% class label of the multi-class. 

% Return the index of p instead of the max_values because max_values is the class
% label which has value of 1  (others being 0), but it is the index of the 
% value which denotes the class label. E.g. index 3 is class 3. 
[max_values,indices] = max(A3, [], 2) 
p = indices
% =========================================================================


end
