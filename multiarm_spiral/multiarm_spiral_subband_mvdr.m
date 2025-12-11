N = 128;    % 阵元数
% nArms = 6;  % 螺旋臂数
% a = 0.01; 
% b = 0.012     % Archimedean spiral parameters r = a + b*theta
fs = 8000 % 采样率 (Hz)
T = 1.0;    % 信号时长 (s)
t = (0:1/fs:T).';
fc = 16000   % 载波 / 中心频率（子带 MVDR 使用）
numSubbands = 128; % MVDR子带数量
speedSound = 343;   % 声速
desiredAngle = [20;0];  % 期望信号入射方向
interfAngles = [-40 60 ; 0 0];   % 干扰方向，按列表示 [az1 az2; el1 el2]
% 信号设置：宽带线性调频 (LFM)
f1 = 500; 
f2 = 3500     % LFM 起止频率（Hz）
% 干扰：窄带正弦 (可多个频率)
interfFreqs = [2500, 2800];

rng(0); % 固定随机种子，便于复现

%% ------------------ 构建阵列坐标 ------------------

xlsxfile = 'D:/matlab_project/beamformer/4-MEMS128坐标.xlsx';
sheetname = 'Sheet1';
% 1) 读取表格
A = readtable(xlsxfile, 'Sheet', 'Sheet1', 'VariableNamingRule', 'preserve');
A.Properties.VariableNames   % 查看实际变量名
% 检查常见列名
if any(strcmp(A.Properties.VariableNames,'PCB-X')) && ...
    any(strcmp(A.Properties.VariableNames,'PCB-Y'))
    xs = A.('PCB-X');
    ys = A.('PCB-Y');
else
    error('未找到 PCB-X / PCB-Y 列，请检查表头或把坐标列重命名为PCB-X, PCB-Y');
end

% 2) 基本统计，供确认（显示）
xmin = min(xs);
xmax = max(xs);
xmean = mean(xs);
ymin = min(ys);
ymax = max(ys);
ymean = mean(ys);
fprintf(['坐标统计 (原始表格单位未知):\n  X: min=%.3f  max=%.3f  mean=%.3f' ...
    '\n  Y: min=%.3f  max=%.3f  mean=%.3f\n'],xmin,xmax,xmean,ymin,ymax,ymean')

% 3) 自动单位判定：若绝对坐标范围 > 10 则很可能是 mm。转换为m
maxRange = max([abs([xmin xmax ymin ymax])]);
if maxRange > 10
    % fprintf('检测到坐标量级较大 (>%g)。假定为 mm，自动转换为 m（除以 1000）。\n',10);
    xs = xs / 1000;
    ys = ys / 1000;
% else
%     % fprintf('坐标量级较小，保留原单位 (假定已为米 m)\n');
end

% 4) 构建 3xN 位置矩阵 (z = 0)
N = numel(xs);
pos = [xs(:)'; ys(:)';zeros(1,N)];

% 可视化阵列布局
figure;
scatter(pos(1,:),pos(2,:),40,1:N,'filled');
axis equal;
grid on;
xlabel('x(m)');
ylabel('y(m)');
title('Spiral array geometry (top view)');
colorbar;
colormap(jet);

figure;
scatter3(pos(1,:), pos(2,:), pos(3,:), 40, (1:N),'filled');
xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
title('3D MEMS Array Layout');
axis equal;
grid on;
hold on;
for i = 1:N
    text(pos(1,i),pos(2,i),pos(3,i),num2str(i),'FontSize',8,'Color','k');
end
hold off;
%% ---------创建阵列对象-----------
mic = phased.OmnidirectionalMicrophoneElement('FrequencyRange',[20 20000]);
array = phased.ConformalArray('Element',mic,'ElementPosition',pos);

%% ------------------ 生成源信号与干扰 ------------------
% 目标信号：基带 LFM，然后调制（wideband collector 用 modulated input）
sig_desired = chirp(t,f1,T,f2);
% 调制到载波 (phased.WidebandCollector 里假定 modulated input true)
% 干扰信号（若多个）
sig_interf = zeros(length(t),size(interfAngles,2));
for k = 1:size(interfAngles,2)
    f0 = interfFreqs(min(k,length(interfFreqs)));
    sig_interf(:,k) = sin(2*pi*f0*t);
end
%% ------------------ 收集到阵列 (WidebandCollector) ------------------
collector = phased.WidebandCollector(...
    'Sensor',array,...
    'PropagationSpeed',speedSound,...
    'SampleRate',fs,...
    'ModulatedInput',true,...
    'CarrierFrequency',fc,...
    'NumSubbands',1024);     % 子带数用于建模

% 目标到达阵列
sig_desired_array = collector(sig_desired, desiredAngle);

% 干扰到达阵列（若多干扰，叠加）
sig_interf_array = zeros(size(sig_desired_array));
for k = 1:size(interfAngles, 2)
    tmp = collector(sig_interf(:,k), interfAngles(:,k));
    sig_interf_array = sig_interf_array + tmp;
end

%% ------------------ 添加噪声 (SNR 调节) ------------------
snr_db = 20;    % 信噪比 dB
signalPower = mean(abs(sig_desired_array(:)).^2);
noisePower = signalPower / (10^(snr_db/10));
noise = sqrt(noisePower/2) * (randn(size(sig_desired_array)) + 1j*randn(size(sig_desired_array)));

% 接收信号 = 期望 + 干扰 + 噪声
rx = sig_desired_array + sig_interf_array + noise;

%% ------------------ 子带 MVDR 波束形成器 (训练使用干扰+噪声) ------------------
bf = phased.SubbandMVDRBeamformer(...
    'SensorArray',array,...
    'Direction',desiredAngle,...
    'OperatingFrequency',fc,...
    'PropagationSpeed',speedSound,...
    'SampleRate',fs,...
    'NumSubbands',numSubbands,...
    'TrainingInputPort',true,...
    'WeightsOutputPort',true,...
    'SubbandsOutputPort',true);

% 用干扰+噪声作为训练数据（让 MVDR 对干扰建协方差），注意训练信号维度应与 rx 匹配
training = sig_interf_array + noise;
[y, W, subbandfreq] = bf(rx, training); % y: beamformed output, W: weights (M x S)

%% ------------------ 时域对比（部分区间） ------------------
figure;
idx = 1:2000;
plot(t(idx)*1000, real(rx(idx, round(N/2))),'r:');
hold on;
plot(t(idx)*1000, real(y(idx)),'b');
xlabel('Time (ms)');
ylabel('Amplitude');
legend('Sample channel', 'Subband-MVDR output');
title('Time-domain comparison (partial)');


%% ------------------ 频谱 / 频域对比（目标频段） ------------------
% 选取较短片段计算频谱对比
seg = 1:4096;
figure;
subplot(2,1,1)
pwelch(real(rx(seg,round(N/2))), hamming(1024),512,1024,fs);
title('PSD: single element (before BF)');
subplot(2,1,2)
pwelch(real(y(seg)), hamming(1024),512,1024,fs);
title('PSD: subband MVDR output (after BF)');

%% ------------------ 子带方向图：选取接近干扰频率的若干子带绘图 ------------------
% 选择几个子带索引（靠近干扰频率）
% subbandfreq 为 1 x S 向量
% 找到与干扰频率最接近的子带
fd = interfFreqs;
[subband_vals, idxs] = arrayfun(@(f) find(abs(subbandfreq - f) ...
    == min(abs(subbandfreq -f)), 1),fd);

figure;
for k = 1:length(idxs)
    idx_fb = idxs(k);
    w_equiv = W(:,idx_fb);
    subplot(1,length(idxs),k);
    pattern(array, subbandfreq(idx_fb),-180:180,0,...
        'PropagationSpeed', speedSound, ...
        'Weights', w_equiv, ...
        'CoordinateSystem','rectangular');
    title(sprintf('Pattern @ %.0f Hz (subband)', subbandfreq(idx_fb)));
end

%% ------------------ 可视化权重幅相 ------------------
figure;
mag = abs(W);
ph = angle(W);
subplot(2,1,1);
imagesc(subbandfreq, 1:N, mag.');
axis xy;
xlabel('Freq (Hz)');
ylabel('Element');
title('Weight magnitude');

subplot(2,1,2);
imagesc(subbandfreq, 1:N, ph.');
axis xy;
xlabel('Freq (Hz)');
ylabel('Element');
title('Weight phase (rad)');



