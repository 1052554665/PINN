clear; clc;

%% ===============================
% 1. 生成合成信号（含噪声）
%% ===============================
N  = 3000;
Ts = 1e-5;
t  = (0:N-1)*Ts;

% 信号 x1(t)
T1 = 211 * 1e-5;
x1 = 5e-3 * sin(2*pi*t / T1);

% 信号 x2(t)
x2 = zeros(size(t));
x2(2000) = 0.8;

% 信号 x3(t)
T3 = 51 * 1e-5;
x3 = zeros(size(t));
x3(2500+1:2600) = 2e-2*sin(2*pi*t(2501:2600)/T3);

% 合成信号
x = x1 + x2 + x3;

% 添加 40 dB 噪声
x = awgn(x, 40, 'measured');

%% ===============================
% 2. STFT-FD 参数
%% ===============================
fs = 1/Ts;
NC = 4;   % 每个频率 4 个周期窗口

% 定义频率轴 (论文常用 0~fs/2)
F = linspace(10, fs/2, 256);  
NF = length(F);

% 时间位置
Tpos = 200 : 20 : 2800;  
NT = length(Tpos);

STFT_FD = zeros(NF, NT);

%% ===============================
% 3. 逐频率计算频率依赖窗长并计算 DFT 第 (1+NC) 项
%% ===============================
for fi = 1:NF
    f = F(fi);

    % ===== 频率依赖窗长 (样本数) =====
    NW = round(NC * fs / f);   

    if mod(NW,2)==1
        NW = NW+1;   % 必须偶数长度，方便对称
    end

    % ===== 生成对称 Hamming 窗 =====
    w = hamming(NW, "symmetric")';

    % ===== 半窗长度 =====
    L = NW/2;

    % ===== 遍历所有时间点 =====
    for ti = 1:NT
        t0 = Tpos(ti);

        % 窗区间
        idx1 = t0 - L + 1;
        idx2 = t0 + L;

        if idx1 < 1 || idx2 > N
            STFT_FD(fi, ti) = 0;
            continue;
        end

        % 取窗内信号
        x_win = x(idx1:idx2);

        % 加窗
        wx = x_win .* w;

        % ====== 仅计算 DFT 的第 (1 + NC) 项 (论文公式 5、6) ======
        k = 1 + NC;
        n = 1:NW;
        Xk = sum(wx .* exp(-1j*2*pi*(k-1)*(n-1)/NW));

        STFT_FD(fi, ti) = Xk;
    end
end

%% ===============================
% 4. 绘制时频图（幅度谱）
%% ===============================
figure;
imagesc(Tpos*Ts, F, abs(STFT_FD));
axis xy;
xlabel("Time (s)");
ylabel("Frequency (Hz)");
title("STFT-FD Time-Frequency Representation (NC=4 cycles)");
colorbar;
colormap jet;
