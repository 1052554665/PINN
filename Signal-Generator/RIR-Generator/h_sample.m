% 设置全局字体
% set(groot,'defaultAxesFontName','SimHei');  % 黑体
% 或使用其他中文字体
% set(groot,'defaultAxesFontName','Microsoft YaHei');  % 微软雅黑

c = 340;    
fs = 16000;
r = [2 1.5 2];  % 接收器位置 [ x y z ] （m）
s = [2 3.5 2];  % 源位置 [ x y z ] （m）
L = [5 4 6];    % 房间尺寸 [ x y z ] （米）
beta = 0.4; % 混响时间（秒）。数值越大，房间反射越强
nsample = 4096; % 采样数 生成的脉冲响应长度=nsample/fs=4096/16000=0.256
% 麦克风的指向性类型；超心形麦克风在前方有较强灵敏度，背部有一定抑制效果，适用于有反射声的场景
% 'omnidirectional'：全向；'cardioid'：心形；'hypercardioid'：超心形；'bidirectional'：8 字形
mtype = 'hypercardioid';    
% 反射阶数，控制房间反射建模深度
% 若为正整数，如 2，则计算直达声 + 一、二阶反射；
% 若为 -1，则自动计算到最大可能反射阶数（直到能量衰减到很低）
order = -1; % −1等于最大反射阶数
dim = 3;    % 房间尺寸
% 第一个参数：方位角（azimuth），即在水平面上相对于 x 轴的角度；
% 第二个参数：仰角（elevation），即相对于水平面的倾斜角
orientation = [pi/2 0]; % 麦克风朝向正 y 方向
hp_filter = 1;  % 开启高通滤波器，滤除直流分量与低频漂移，模拟真实麦克风响应
% 该函数生成从声源 s 到麦克风 r 的 房间脉冲响应 h
h = rir_generator(c, fs, r, s, L, beta, nsample, mtype, order, dim, orientation, hp_filter);

%% 查看RIR
figure(1);
plot((0:nsample-1)/fs, h, 'b','LineWidth',1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Room Impulse Response');
grid on;
exportgraphics(gcf,'1.png','Resolution',600);

%% 生成原始正弦信号
t = (0:1/fs:1)';
f0 = 440;
x = sin(2*pi*f0*t);

%% 卷积加入混响
y = conv(x, h);
y = y / max(abs(y));    % 归一化
x = x / max(abs(x));    % 同样归一化便于比较

figure(2);
% figure('Name','波形对比');

% 绘制波形对比
t_x = (0:length(x)-1) / fs;
t_y = (0:length(y)-1) / fs;

subplot(2,1,1)
plot(t_x, x,'b');
xlabel('Time (s)');
ylabel('Amplitude');
title('原始正弦信号波形')
grid on;

subplot(2,1,2);
plot(t_y, y, 'r');
xlabel('Time (s)');
ylabel('Amplitude');
title('加入混响后的信号波形');
grid on;

exportgraphics(gcf,'2.png','Resolution',600);

figure(3);
% figure('Name','频谱对比');

% 绘制频谱对比
N_x = length(x);
N_y = length(y);
f_x = (0:N_x-1)*(fs/N_x);
f_y = (0:N_y-1)*(fs/N_y);

% 取单边频谱
X = abs(fft(x));
Y = abs(fft(y));

subplot(2,1,1);
plot(f_x(1:floor(N_x/2)), 20*log10(X(1:floor(N_x/2))/max(X)),'b');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('原始信号幅频图');
xlim([0 fs/2]);
grid on;

subplot(2,1,2);
plot(f_y(1:floor(N_y/2)), 20*log10(Y(1:floor(N_y/2))/max(Y)),'r');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('加入混响后的幅频图');
xlim([0 fs/2]);
grid on;

exportgraphics(gcf,'3.png','Resolution',600);

figure(4);
subplot(2,1,1);
spectrogram(x, hamming(256), 128, 512, fs, 'yaxis');
title('原始正弦信号频谱图');

subplot(2,1,2);
spectrogram(y, hamming(256), 128, 512, fs, 'yaxis');
title('加入混响后的频谱图');

colormap jet;
exportgraphics(gcf, '4.png', 'Resolution', 600);