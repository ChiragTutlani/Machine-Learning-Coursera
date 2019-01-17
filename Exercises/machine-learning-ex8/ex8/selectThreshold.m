function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
epsilon = min(pval):stepsize:max(pval);
F1 = zeros(length(epsilon),2);

for i=1:length(epsilon)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

predictions = pval < epsilon(i);
truePositive = sum((yval == 1)&(predictions == 1));
falsePositive = sum((yval == 0)&(predictions == 1));
falseNegative = sum((yval == 1)&(predictions == 0));

precision = truePositive/(truePositive+falsePositive);
recall = truePositive/(truePositive+falseNegative);
score = (2*precision*recall)/(precision+recall);
F1(i,:) = [epsilon(i),score];


end

[bestF1 index]=max(F1(:,2));
bestEpsilon = F1(index,1);
end
