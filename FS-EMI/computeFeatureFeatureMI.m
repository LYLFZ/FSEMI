function I = computeFeatureFeatureMI(X)
    % 计算特征矩阵 X 中每对特征之间的互信息
    % 输入参数：
    % X: 特征矩阵，大小为 (m, n)
    % 输出参数：
    % I: 互信息矩阵，大小为 (n, n)

    % 特征数
    [m, n] = size(X);

    % 初始化互信息矩阵
    I = zeros(n, n);

    % 计算每对特征之间的互信息
    for i = 1:n
        for j = i+1:n
            I(i, j) = mutualInformation(X(:, i), X(:, j));
            I(j, i) = I(i, j); % 互信息矩阵是对称的
        end
    end
end

function MI = mutualInformation(X, Y)
    % 计算向量 X 和 Y 之间的互信息
    % 输入参数：
    % X: 向量 X
    % Y: 向量 Y
    % 输出参数：
    % MI: 互信息

    % 联合直方图
    jointHist = histcounts2(X, Y, 'Normalization', 'probability');

    % 边缘直方图
    pX = sum(jointHist, 2);
    pY = sum(jointHist, 1);

    % 计算互信息
    MI = 0;
    for i = 1:length(pX)
        for j = 1:length(pY)
            if jointHist(i, j) > 0
                MI = MI + jointHist(i, j) * log2(jointHist(i, j) / (pX(i) * pY(j)));
            end
        end
    end
end
