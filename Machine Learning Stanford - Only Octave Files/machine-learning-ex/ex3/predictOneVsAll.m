function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
probabilities = sigmoid(X*all_theta') % note all_theta is a c*n matrix where
                                      % c is number of classes. So probabilities     
                                      % is a m*c matrix
                                      
[max_values,indices] = max(probabilities, [], 2) % Take the max_value of each row
                                                 % of the matrix 
                                               
p = indices % Note the decision rule for multiclass classification is just to choose
            % The class with the highest probability compared to other classes    
            % E.g. If a point x has [class1 = 0.48, class2=0.45 class3 = 0.42]
            % Then the choice is 0.48. So there is no threshold value to satisfy,
            % Since I want only one class to be chosen. 

% =========================================================================


end
