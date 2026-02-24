function [modProb,sumProb] = predict(features,weights)
[~,c] = size(weights);
modProb = exp(features * weights);
if sum(sum(isinf(modProb)))
    loc = isinf(modProb);
    modProb(loc) = realmax('double');
end
sumProb = sum(modProb,2);   
modProb = modProb./repmat(sumProb,[1,c]);
modProb(modProb==0)=realmin('double');
end

