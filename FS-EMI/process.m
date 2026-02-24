function [target,gradient] = process(weights)

global features labels G c1 c2 c3 c4 FFMI FLMI

[row,cow]=size(weights);
    modProb = exp(features * weights);
    if sum(sum(isinf(modProb)))
    loc = isinf(modProb);
    modProb(loc) = realmax('double');
    end
    sumProb = sum(modProb,2);   
    modProb = modProb./repmat(sumProb,[1,cow]);
    modProb(modProb==0)=realmin('double');                                    
    costfir=-sum(sum(labels.*log(modProb)));
    D = eye(row)./(2*sqrt(weights*weights')+1e-5);
    costsec = trace(weights'*D*weights);
    costthre=norm(modProb*G-modProb,'fro')^2;
    costfour = sum(FLMI .* sqrt(sum(weights.^2, 2)));
    costfive = 0;
    [m, ~] = size(weights);
    for i = 1:m
        for j = 1:m
            if i ~= j
                costfive = costfive + FFMI(i, j) * norm(weights(i,:) - weights(j,:), 2);
            end
        end
    end
    target =costfir+2*c1*costsec+c2*costthre+c3*costfour-c4/row*costfive;
    eye_I = eye(size(G,1));
    grad1=features'*(modProb - labels);
    grad2=(D*weights);
    [n,d] = size(features);
    c = size(labels,2);
    H=G-eye_I;
    temp = modProb*H; 
    H_sub_temp = zeros(n,c);
    for z=1:c   
    H_sub_temp(:,z) = sum(temp.*(repmat(H(z,:),[n,1])-temp),2);
    end
    grad3 = features'*(modProb.*H_sub_temp); 
    grad4 = zeros(size(weights));
    W_norm = sqrt(sum(weights.^2, 2));
    grad4 = c3 * bsxfun(@times, FLMI, weights ./ W_norm);
    grad5 = zeros(size(weights));
    W_diff = permute(weights, [1 3 2]) - permute(weights, [3 1 2]);
    W_diff_norm = sqrt(sum(W_diff.^2, 3));
    W_diff_norm(W_diff_norm == 0) = 1;
    I_X_X_rep = repmat(FFMI, [1 1 cow]);
    grad5 = -c4/row * sum(bsxfun(@times, I_X_X_rep, W_diff ./ W_diff_norm), 2);
    grad5 = reshape(grad5, [row cow]);
    gradient =grad1+2*c1.*grad2+2*c2.*grad3 + grad4 + grad5;
end

