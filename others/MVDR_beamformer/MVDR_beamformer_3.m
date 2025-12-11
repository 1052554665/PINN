% ======================================================
% 1️⃣ 加载你的真实“目标信号”和“干扰信号”
% ======================================================
% [targetSig, fs1]   = audioread('DCBias2_60.wav');      % ✅ 你的目标信号
% [targetSig, fs1]   = audioread('G4_10pSeventhHarmonic_55.wav');      % ✅ 你的目标信号
% [targetSig, fs1]   = audioread('Loosen1_60.wav');      % ✅ 你的目标信号
[targetSig, fs1]   = audioread('Normal_part92.wav');      % ✅ 你的目标信号
% [targetSig, fs1]   = audioread('PartialDischarge1_60.wav');      % ✅ 你的目标信号


[interfSig, fs2]   = audioread('振安1#反_part46.wav'); % ✅ 你的干扰信号

targetSig  = targetSig(:,1);
interfSig  = interfSig(:,1);

% 采样率统一
if fs1 ~= fs2
    error('目标信号与干扰信号采样率不一致！');
end
fs = fs1;

% 长度对齐
L = min(length(targetSig), length(interfSig));
targetSig  = targetSig(1:L);
interfSig  = interfSig(1:L);

% 归一化
targetSig  = targetSig / max(abs(targetSig));
interfSig  = interfSig / max(abs(interfSig));

t = (0:L-1)' / fs;

% ======================================================
% 2️⃣ 阵列参数
% ======================================================
c  = 340;
fc = 300e6;

array = phased.ULA( ...
    'NumElements', 5, ...
    'ElementSpacing', 0.5);

% ======================================================
% 3️⃣ 双入射角（目标 + 干扰）
% ======================================================
targetAngle    = [45; 0];     % ✅ 目标方向
interfereAngle = [-25; 0];    % ✅ 干扰方向（可自行修改）

% ======================================================
% 4️⃣ 阵列接收：目标 + 干扰
% ======================================================
x_target  = collectPlaneWave(array, targetSig,  targetAngle,   fc, c);
x_interf  = collectPlaneWave(array, interfSig,  interfereAngle,fc, c);

x_total = x_target + x_interf;

% 加噪声（可调）
noise = 0.03 * (randn(size(x_total)) + 1j*randn(size(x_total)));
rx = x_total + noise;

% ======================================================
% 5️⃣ MVDR 波束形成（对准目标）
% ======================================================
beamformer = phased.MVDRBeamformer( ...
    'SensorArray', array, ...
    'PropagationSpeed', c, ...
    'OperatingFrequency', fc, ...
    'Direction', targetAngle, ...
    'WeightsOutputPort', true);

[y, w] = beamformer(rx);

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