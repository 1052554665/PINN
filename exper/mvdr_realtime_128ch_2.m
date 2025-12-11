function udp_128ch_mvdr_realtime()

%% ============================================================
%                      基本参数
% ============================================================
localIP  = "192.168.0.3";
localPort = 8080;

% 定义频谱图参数
specWin   = 512;      % STFT window
specHop   = 400;
specNfft  = 512;
specMaxF  = 8000;     % 上限 8kHz
outputDir = "mvdr_output";  % 输出文件夹
if ~exist(outputDir, 'dir'); mkdir(outputDir); end

expectedHeader = uint32(hex2dec('EB9055CC'));
numChannels = 128;
framesPerPacket = 4;
bytesPerPacket  = 1040;
payloadBytes    = 1024;

Fs = 192000;
displayChannel = 1;          % 只显示一个通道

timeWin = 2048;              % FFT 窗长度
fftUpdateInterval = 0.05;    % 0.05 秒更新一次 FFT
specUpdateInterval = 0.2;    % 0.2 秒更新一次谱图
maxFreq = 8000;              % 频率上限 8 kHz
saveInterval = 1.0;          % WAV 每 1 秒保存一次

fprintf("=============== MVDR Real-Time System Start ===============\n");

%% ============================================================
%                麦克风几何结构 &  Steering Vector
% ============================================================
data = readmatrix('mic_positions.xlsx');
X = data(:,1); 
Y = data(:,2);
Z = zeros(size(X));

mic_pos = [X Y Z] / 1000;    % mm → m
Nmic = size(mic_pos,1);

if Nmic ~= numChannels
    error("Mic count mismatch: Excel=%d, UDP=%d", Nmic, numChannels);
end

c = 340;
target_pos = [2, 3.5, 4];     % 真实环境使用 DOA 后替换

array_center = mean(mic_pos,1);
steer_vec = (target_pos - array_center);
steer_vec = steer_vec / norm(steer_vec);

tau = (mic_pos - array_center) * steer_vec' / c;
tau = tau(:);   % 强制 128×1

freqs = (0:timeWin/2-1) * Fs/timeWin;   % 1×2048

steeringFreq = exp(-1i * 2*pi * (tau * freqs));   % 128×2048


%% ============================================================
%                       GUI 界面
% ============================================================
hFig = figure('Name','UDP + MVDR Realtime','NumberTitle','off',...
              'Position',[100 100 1200 900]);

subplot(2,2,1);
hTime = plot(zeros(timeWin,1)); grid on; axis tight;
title("Channel Time Domain");

subplot(2,2,2);
hFFT = plot(zeros(timeWin/2,1)); grid on; axis tight;
title("Channel FFT");

% subplot(3,2,[3 4]);
% hMVDR = plot(zeros(timeWin/2,1)); grid on; axis tight;
% title("MVDR Output FFT");

subplot(2,2,[3 4]);
hSpec = imagesc([],[],zeros(1024,200));
axis xy; caxis auto; colorbar; title("MVDR Spectrogram");
colormap jet;


%% ============================================================
%                       UDP 初始化
% ============================================================
u = udpport("LocalHost",localIP,"LocalPort",localPort,"Timeout",0.5);
fprintf("UDP started at %s:%d\n",localIP,localPort);

buf = zeros(timeWin,1);                   % FFT buffer
spectrogram_buf = [];                     % MVDR spectrogram
fftTimer = tic;
specTimer = tic;

% ========= MVDR R_xx =========
alpha = 0.95;
Rxx = eye(numChannels) * 1e-3;
Rinv = inv(Rxx);

%% ============================================================
%     保存 1 秒的 MVDR 音频 + 频谱图（同步保存）
% ============================================================
mvdrBuf = zeros(Fs,1);         % 1秒=192k样本
mvdrCounter = 0;
wavIndex = 1;
lastSaveTime = tic;

%% ============================================================
%                       处理循环
% ============================================================
while ishandle(hFig)

    raw = read(u, bytesPerPacket, "uint8");
    if numel(raw) ~= bytesPerPacket
        continue;
    end

    % ------ 提取 PCM ------
    pcm = typecast(uint8(raw(17:16+payloadBytes)), 'int16');
    pcm = double(reshape(pcm, numChannels, framesPerPacket));

    x = pcm(:,1);   % 只取第 1 帧

    % =============== 1) 更新 FFT buffer ===============
    buf(1:end-1) = buf(2:end);
    buf(end) = x(displayChannel);

    % =============== 2) 更新 Rxx ===============
    Rxx = alpha*Rxx + (1-alpha)*(x*x');
    Rinv = inv(Rxx + 1e-6 * eye(numChannels));

    % =============== 3) MVDR 输出 ===============
    % w = Rinv * a / (a' * Rinv * a)
    a = steeringFreq(:,1);   % 使用第一个频点方向，简化计算

    w = (Rinv * a) / (a' * Rinv * a);
    y_mvdr = real(w' * x);

%% =============== 保存 MVDR 音频与频谱图（每 1 秒） ===============
    mvdrCounter = mvdrCounter + 1;
    
    if mvdrCounter <= Fs
        mvdrBuf(mvdrCounter) = y_mvdr;
    end
    
    % ---- 判断 1 秒是否到达（严格按时间） ----
    if toc(lastSaveTime) >= 1.0
    
        %% ---- 音频不足补 0 ----
        if mvdrCounter < Fs
            mvdrBuf(mvdrCounter+1:Fs) = 0;
        end
    
        %% ---- 保存 MVDR 音频 ----
        audioOut = mvdrBuf / max(abs(mvdrBuf) + 1e-9);
        wavName = sprintf("mvdr_%04d.wav", wavIndex);
        audiowrite(fullfile(outputDir, wavName), audioOut, Fs);
        fprintf("Saved WAV: %s\n", wavName);
    
        %% ---- 计算 MVDR 的频谱图 ----
        [S, F, T] = spectrogram(audioOut, hann(specWin), specHop, specNfft, Fs);
        idxF = F <= specMaxF;
        Smag = 20*log10(abs(S(idxF,:)) + 1e-8);
    
        %% ---- 保存频谱图 PNG ----
        fig = figure('Visible','off','Position',[100 100 1200 600]);
        imagesc(T, F(idxF), Smag);
        axis xy;
        xlabel("Time (s)");
        ylabel("Frequency (Hz)");
        title(sprintf("MVDR Spectrogram %04d", wavIndex));
        colormap jet; colorbar;
    
        pngName = sprintf("mvdr_%04d.png", wavIndex);
        saveas(fig, fullfile(outputDir, pngName));
        close(fig);
        fprintf("Saved PNG: %s\n", pngName);
    
        %% ---- 重置计数 ----
        mvdrCounter = 0;
        wavIndex = wavIndex + 1;
        lastSaveTime = tic;
    end

    % =============== 4) 更新 GUI（降低刷新频率） ===============
    if toc(fftTimer) > fftUpdateInterval
        fftTimer = tic;
        Xk = fft(buf .* hann(timeWin), timeWin);
        P = abs(Xk(1:timeWin/2));
        set(hFFT, "YData", 20*log10(P+1e-6));
        set(hTime,"YData",buf);
    end

    if toc(specTimer) > specUpdateInterval
        specTimer = tic;
        [S,F,T] = spectrogram(buf, hann(512), 400, 512, Fs);
        idx = F <= maxFreq;
        set(hSpec,'CData',20*log10(abs(S(idx,:))+1e-6));
        set(hSpec,'YData',F(idx));
    end

    drawnow limitrate nocallbacks;

end

end
