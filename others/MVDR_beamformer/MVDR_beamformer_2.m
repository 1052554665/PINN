% ======================================================
% 5元均匀线阵 (ULA) 的 MVDR 波束形成演示
% ======================================================
clc; clear; close all;

% 参数设置
t = [0:0.1:500]';
fr = 1000;              % 信号频率 (Hz)
xm = sin(2*pi*fr*t);    % 输入信号
c = physconst('LightSpeed');
fc = 300e6;             % 载波频率
rng('default');
incidentAngle = [45; 0];  % 入射角 [方位角; 仰角]

% 阵列定义
array = phased.ULA('NumElements',5,'ElementSpacing',0.5);

% 接收信号
x = collectPlaneWave(array,xm,incidentAngle,fc,c);
noise = 0.1*(randn(size(x)) + 1j*randn(size(x)));
rx = x + noise;

% MVDR 波束形成
beamformer = phased.MVDRBeamformer( ...
    'SensorArray',array, ...
    'PropagationSpeed',c, ...
    'OperatingFrequency',fc, ...
    'Direction',incidentAngle, ...
    'WeightsOutputPort',true);

[y, w] = beamformer(rx);  % 输出波束形成信号和权重


% ======================================================
% IEEE 单栏：时域 + STFT 上下双图 + SNR 自动标注 + PDF
% 图宽 = 8.5 cm = 3.35 inch
% ======================================================

% --------- 1. 计算输入 & 输出 SNR ---------
signal_clean = x(:,3);        % 无噪原始信号（第3阵元）
signal_noisy = rx(:,3);       % 含噪信号
signal_mvdr  = y;             % MVDR 输出

noise_in  = signal_noisy - signal_clean;
noise_out = signal_mvdr  - signal_clean;

SNR_in  = 10*log10( var(real(signal_clean)) / var(real(noise_in)) );
SNR_out = 10*log10( var(real(signal_clean)) / var(real(noise_out)) );
SNR_gain = SNR_out - SNR_in;

snr_text = sprintf('SNR_{in}=%.2f dB   SNR_{out}=%.2f dB   Gain=%.2f dB', ...
                     SNR_in, SNR_out, SNR_gain);

% --------- 2. IEEE 单栏尺寸设定 ---------
figWidth  = 3.35;   % inch  (8.5 cm)
figHeight = 4.8;    % inch  (上下双图比例)

figure('Units','inches','Position',[2 2 figWidth figHeight],'Color','w');

% ======================================================
% (a) 上图：时域对比
% ======================================================
subplot(2,1,1);

plot(t, real(signal_noisy), 'r--', 'LineWidth', 0.9); hold on;
plot(t, real(signal_mvdr),  'b',   'LineWidth', 1.2);

ylabel('Amplitude', 'FontSize', 9);
title('(a) Time-Domain Signal Comparison', 'FontSize', 10, 'FontWeight', 'bold');
legend('Original','MVDR Output','FontSize',8,'Location','northeast');
grid off;
box on;
set(gca,'FontSize',9,'LineWidth',0.9);

% --- SNR 标注（自动）---
text(0.01, 0.08, snr_text, 'Units','normalized', ...
    'FontSize', 8, 'FontWeight','bold');

% ======================================================
% (b) 下图：STFT（MVDR 输出）
% ======================================================
subplot(2,1,2);

fs = 1 / (t(2) - t(1));
window = hamming(128);
noverlap = 64;
nfft = 256;

spectrogram(real(signal_mvdr), window, noverlap, nfft, fs, 'yaxis');
title('(b) STFT of MVDR Output', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('Time (s)', 'FontSize', 9);
ylabel('Frequency (Hz)', 'FontSize', 9);
set(gca,'FontSize',9,'LineWidth',0.9);
colormap turbo;
colorbar;

% ======================================================
% 3. 保存为 IEEE 期刊级 PDF（矢量）
% ======================================================
exportgraphics(gcf, 'IEEE_Time_STFT_SNR_MVDR.pdf', 'ContentType','vector');















% ======================================================
% 1. 原始信号 vs MVDR 输出（同图对比 + PDF 保存）
% ======================================================

figure('Color','w','Position',[200 200 900 350]);

plot(t, real(rx(:,3)), 'r--', 'LineWidth', 1.2); hold on;
plot(t, real(y),       'b',   'LineWidth', 1.4);

xlabel('Time (s)', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 18, 'FontWeight', 'bold');
legend('Original Received Signal','MVDR Output', ...
       'FontSize', 16, 'Location', 'best');

title('Time-Domain Comparison of Original and MVDR Output', ...
      'FontSize', 18, 'FontWeight', 'bold');

set(gca, 'FontSize', 15, 'LineWidth', 1.2);
grid off;
box on;

% ===== 保存为 PDF（矢量图，适合论文）=====
exportgraphics(gcf, 'Raw_vs_MVDR_TimeDomain.pdf', 'ContentType','vector');




% ======================================================
% 2. 2D 波束方向图（水平面）
% ======================================================
figure;
pattern(array, fc, -180:180, 0, ...
    'PropagationSpeed', c, ...
    'Weights', w, ...
    'CoordinateSystem', 'rectangular', ...
    'Type', 'powerdb');
title('2D Beam Pattern (Azimuth Plane)', 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'FontSize', 13, 'LineWidth', 1.2);

% ======================================================
% 3. 三维波束方向图（方位角 + 仰角）
% ======================================================
figure;
pattern(array, fc, ...
    -180:180, -90:90, ...  % 方位角范围 -180~180°, 仰角 -90~90°
    'PropagationSpeed', c, ...
    'Weights', w, ...
    'CoordinateSystem', 'polar', ...
    'Type', 'powerdb');
title('3D MVDR Beam Pattern', 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'FontSize', 13);
view(45, 30);  % 调整3D视角


% ======================================================
% 4. 极坐标波束图（Polar Plot）
% ======================================================

% 扫描角度范围（水平面 -90°~90°）
az = -90:0.5:90;
el = 0;

% 计算方向图增益 (dB)
patternData = pattern(array, fc, az, el, ...
    'PropagationSpeed', c, ...
    'Weights', w, ...
    'Type', 'powerdb');

% 绘制极坐标波束图
figure('Color', 'w');
p = polarplot(deg2rad(az), patternData, 'k', 'LineWidth', 2);

% 图形样式设置
ax = gca;
ax.ThetaZeroLocation = 'top';      % 0° 在上方
ax.ThetaDir = 'clockwise';         % 顺时针方向
ax.RLim = [-60 0];                 % dB 范围
ax.ThetaTick = -90:30:90;          % 角度刻度
ax.RTick = -60:10:0;               % 径向刻度
ax.FontSize = 16;
ax.LineWidth = 1.2;

title('MVDR Beam Pattern (Polar Plot)', 'FontSize', 18, 'FontWeight', 'bold');

% 标记目标方向
hold on;
polarplot(deg2rad([incidentAngle(1) incidentAngle(1)]), [-60 0], ...
    'r--', 'LineWidth', 1.5);
legend('Beam Pattern','Target Direction (45°)', ...
    'Location','southoutside','FontSize',16);


% ======================================================
% 4. 短时傅里叶变换 (STFT) 时频分析
% ======================================================

fs = 1 / (t(2) - t(1));  % 根据时间间隔估算采样率
window = hamming(128);   % 窗函数
noverlap = 64;           % 重叠长度
nfft = 256;              % FFT 点数

figure;
spectrogram(real(y), window, noverlap, nfft, fs, 'yaxis');
title('STFT of Beamformed Output', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Frequency (Hz)', 'FontSize', 14, 'FontWeight', 'bold');
colorbar;
set(gca, 'FontSize', 12, 'LineWidth', 1.2);
colormap turbo;  % 更直观的配色
grid on;