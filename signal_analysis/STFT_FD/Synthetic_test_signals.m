%% 参数设置
N  = 3000;        % 样本数
Ts = 1e-5;        % 采样周期
t  = (0:N-1) * Ts; % 时间向量

%% ------------------------
%  x1(t) = 5e-3 * sin( 2πt / (211*1e-5) )
%  周期 T1 = 211 * 1e-5
%% ------------------------
T1 = 211 * 1e-5;
x1 = 5e-3 * sin( 2*pi * t / T1 );

%% ------------------------
%  x2(t) = 0.8  (t = 2000*1e-5)    → 即对应样本 index = 2000
%          0    其他
%% ------------------------
x2 = zeros(size(t));
idx2 = round(2000);   % t = 2000*1e-5 对应第2000个样本
x2(idx2) = 0.8;

%% ------------------------
%  x3(t) = 2e-2*sin(2πt/(51*1e-5))   2500*1e-5 < t ≤ 2600*1e-5
%          0                       其他
%% ------------------------
x3 = zeros(size(t));
T3 = 51 * 1e-5;

start3 = round(2500);  % 对应样本区间 (2500, 2600]
end3   = round(2600);

x3(start3+1:end3) = 2e-2 * sin( 2*pi * t(start3+1:end3) / T3 );

%% 合成信号
x = x1 + x2 + x3;

%% ------------------------
% 添加 40 dB 高斯白噪声
%% ------------------------
% 逐样本 SNR = 40 dB
x_noisy = awgn(x, 40);

%% 绘图

figure(1);
plot(t, x, 'b'); hold on;
% plot(t, x, 'b');
legend('Original signal');
xlabel('Time (s)');
ylabel('Amplitude');
title('Synthetic Signal');


figure(2);
plot(t, x_noisy, 'b'); hold on;
% plot(t, x, 'b');
legend('Noisy signal (40 dB)');
xlabel('Time (s)');
ylabel('Amplitude');
title('Synthetic Signal + 40 dB Gaussian Noise');


