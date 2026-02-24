function I = computeFeatureLabelMI(X, Y)
    % 计算每个特征与标签空间之间的互信息
    % 输入参数：
    % X: 特征矩阵，大小为 (m, n)
    % Y: 标签矩阵，大小为 (m, k)
    % 输出参数：
    % I: 互信息向量，大小为 (n, 1)

    % 特征数和标签数
    [m, n] = size(X);
    [~, k] = size(Y);

    % 初始化互信息向量
    I = zeros(n, 1);

    % 计算每个特征与标签之间的互信息
    for i = 1:n
        mi_sum = 0;
        for j = 1:k
            mi_sum = mi_sum + mutualInformation(X(:, i), Y(:, j));
        end
        % 可以选择取平均或者其他加权方式
        I(i) = mi_sum / k; % 这里取的是平均互信息
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