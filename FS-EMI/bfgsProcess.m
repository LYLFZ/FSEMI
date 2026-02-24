function [target,gradient] = bfgsProcess(weights)

global features_i;
global labels_i;
c = size(labels_i,2);


% lambda=0.5;
modProb = exp(features_i * weights);
if sum(sum(isinf(modProb)))
    loc = isinf(modProb);
    modProb(loc) = realmax('double');
end
sumProb = sum(modProb,2);   
modProb = modProb./repmat(sumProb,[1,c]);
modProb(modProb==0)=realmin('double');

% Target function.
target = -sum(sum(labels_i.*log(modProb)));

% The gradient.
gradient = features_i'*(modProb - labels_i);

end
