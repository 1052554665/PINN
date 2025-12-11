%% 参数设置
L = 16;                  % 滤波器长度
n = 0:L-1;               % 时间索引
w = hamming(L).';        % 汉明窗 (行向量)
w0 = pi/4;               % 中心频率 (可修改)

%% 冲激响应 h[n] = w[n] * exp(j*w0*n)
h = w .* exp(1j * w0 * n);

%% 频率响应
Nfft = 4096;
[H, w_axis] = freqz(h, 1, Nfft, 'whole'); 
w_axis = w_axis - pi;       % 平移到 [-pi, pi]
H = fftshift(H);

%% 绘图 - 幅度响应
figure;
plot(w_axis, 20*log10(abs(H)+eps), 'LineWidth', 2);
xlabel('Frequency (rad/sample)');
ylabel('Magnitude (dB)');
title('Magnitude Response of Bandpass Filter (Hamming, L=16)');
grid on;

%% 绘图 - 相位响应
figure;
plot(w_axis, angle(H), 'LineWidth', 2);
xlabel('Frequency (rad/sample)');
ylabel('Phase (rad)');
title('Phase Response');
grid on;
