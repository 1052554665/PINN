function [TF, F, T] = stft_fd(x, fs, Tpos, FreqList, NC)
% =============================================================
%  Frequency-Dependent STFT (STFT-FD)
%
%  输入：
%     x         : 信号（列向量）
%     fs        : 采样频率
%     Tpos      : 时间中心点索引，例如 200:20:2800
%     FreqList  : 频率列表，例如 linspace(10, fs/2, 256)
%     NC        : 每个频率的周期数（例如 4）
%
%  输出：
%     TF        : 时频矩阵（NF × NT）
%     F         : 频率轴
%     T         : 时间轴
%
% =============================================================

x = x(:);                  % 确保列向量
N = length(x);
F = FreqList(:);
NF = length(F);
T = Tpos(:);
NT = length(T);

TF = zeros(NF, NT);        % 输出矩阵


%% ===============================
% 逐频率计算频率依赖窗长并计算 DFT 第 (1+NC) 项
%% ===============================
% ========== 主循环：逐频率 × 逐时间点 ==========
for fi = 1:NF
    f = F(fi);

    % ====== 计算该频率的窗长（NC 个周期） ======
    % ===== 频率依赖窗长 (样本数) =====
    NW = round(NC * fs / f);

    % 强制偶数长度（必须对称）
    if mod(NW, 2) == 1
        NW = NW + 1;
    end

    L = NW/2;                     % 半窗长
    w = hamming(NW, "symmetric")';% 对称 Hamming 窗
    

    % ====== 对每个时间点计算该频率的 DFT 分量 ======
    for ti = 1:NT
        t0 = T(ti);

        idx1 = t0 - L + 1;
        idx2 = t0 + L;

        % 边界检查
        if idx1 < 1 || idx2 > N
            TF(fi, ti) = 0;
            continue
        end

        % % 取信号 × 窗
        % x_win = x(idx1:idx2)' .* w;

        % 取窗内信号
        x_win = x(idx1:idx2)';

        % 加窗
        wx = x_win .* w;

        % ====== 只计算 DFT 的第 (1+NC) 项（论文关键） ======
        k = 1 + NC; % DFT index
        n = 1:NW;
        Xk = sum( wx .* exp(-1j * 2*pi * (k-1) * (n-1) / NW) );

        TF(fi, ti) = Xk;
    end
end
end
