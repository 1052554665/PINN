function udp_128ch_mvdr_realtime()
% UDP 128ch 实时接收 + 实时 MVDR (无 RIR) + 每秒保存 MVDR spectrogram png
% 说明：适用于 payload 每包 framesPerPacket samples per channel (int16)
% 要求：当前目录含 mic_positions.xlsx (至少 X,Y 列)，UDP 发送到 localIP:localPort

%% =================== 参数 ===================
localIP = "192.168.0.3";
localPort = 8080;

expectedHeader = uint32(hex2dec('EB9055CC'));
numChannels = 128;
framesPerPacket = 4;
bytesPerPacket  = 1040;
payloadBytes    = 1024;   % 1024 bytes -> 512 int16 -> 128 channels x 4 frames

Fs = 192000;
displayChannel = 1;

% 时域 / 频域窗口（可调整以降低 CPU）
timeWin = 2048;                % 改小以降低运算（之前 4096 太大）
fftUpdateInterval = 0.25;      % 每 0.25s 做一次频域 MVDR（可调：增大可降 CPU）

% MVDR 参数
c = 343;
% 目标位置（仅用于 steering vec 的方向估计，单位 m）
target_pos = [2, 3.5, 4];
fprintf("MVDR active. Target pos = [%.2f %.2f %.2f]\n", target_pos);

% MVDR / 数值稳定性
alpha = 0.995;                  % Rxx 指数平滑因子 (0.98~0.999)
diagLoading = 1e-6;             % 对角加载

% spectrogram / 保存设置
enableSpectrogram = true;
maxSpecCols = 200;
saveInterval = 1.0;             % 每 1 秒保存一张频谱图
outputDir = "spectrograms";
if ~exist(outputDir, "dir"), mkdir(outputDir); end

%% =================== 读取麦克风坐标 ===================
data = readmatrix('mic_positions.xlsx');
X = data(:,1); Y = data(:,2);
if size(data,2) >= 3
    Z = data(:,3);
else
    Z = zeros(size(X));
end
mic_pos = [X Y Z] / 1000;   % mm -> m (如原文件单位为 mm)

Nmic = size(mic_pos,1);
if Nmic ~= numChannels
    error("Mic count mismatch: Excel=%d, expected numChannels=%d", Nmic, numChannels);
end
array_center = mean(mic_pos,1);

%% =================== 预计算 steering delays/steering matrix template ===================
vec = target_pos - array_center;
dist = norm(vec);
steer_dir = vec / dist;                   % unit vector
tau = (mic_pos - array_center) * steer_dir' / c;   % Nmic x 1 delays (s)

nFreq = timeWin/2;   % we'll use positive-frequency bins (1..nFreq)
freqs = (0:nFreq-1)' * (Fs / timeWin);        % column

% steering matrix: Nmic x nFreq, complex
steerMat = zeros(Nmic, nFreq);
for k = 1:nFreq
    steerMat(:,k) = exp(-1j*2*pi*freqs(k) * tau);
end

%% =================== GUI 初始化 ===================
hFig = figure('Name','UDP + MVDR Realtime','NumberTitle','off',...
              'Position',[50 50 1200 800]);

subplot(3,2,1);
hTime = plot(zeros(timeWin,1)); grid on; axis tight;
title("Channel Time Domain");

subplot(3,2,2);
hFFT = plot(zeros(nFreq,1)); grid on; axis tight;
title("Channel FFT");

subplot(3,2,[3 4]);
hMVDR = plot(zeros(nFreq,1)); grid on; axis tight;
title("MVDR Output Spectrum");

subplot(3,2,[5 6]);
hSpec = imagesc(zeros(nFreq, maxSpecCols));
axis xy; colorbar; title("MVDR Spectrogram");
xlabel('Frame'); ylabel('Freq bin');

drawnow;

%% =================== UDP 初始化 ===================
u = udpport("LocalHost", localIP, "LocalPort", localPort, "Timeout", 0.5);
fprintf("UDP started at %s:%d\n", localIP, localPort);

%% =================== 缓冲与状态变量 ===================
% multi-channel time window buffer (timeWin x Nmic)
bufCh = zeros(timeWin, Nmic);        % double
% 按帧写入 bufCh：用索引移动（高效）
writeStep = framesPerPacket;         % 每包包含的时间步数

% 协方差矩阵 Rxx 初始化（Nmic x Nmic）
Rxx = eye(Nmic) * 1e-6;             % small diag to avoid singular

frameCount = 0;

% ===== MVDR 输出缓存，用于保存 WAV =====
mvdrBuf = zeros(Fs, 1);   % 1 秒缓冲区
mvdrPtr = 1;              % 写指针
wavIndex = 1;             % 文件编号


fftTimer = tic;
specTimer = tic;
lastSaveTime = tic;

mvdr_spec_data = zeros(nFreq,0);    % will append columns
endianness = "";
detectCount = 0;

%% =================== 主循环 ===================
fprintf("Entering main loop. Press figure close to stop.\n");
while ishandle(hFig)
    % 1) 读 UDP 包
    raw = read(u, bytesPerPacket, "uint8");
    if numel(raw) ~= bytesPerPacket
        pause(0.001);
        continue;
    end

    % 2) 字节序检测（前三包）
    if detectCount < 3
        hdr_be = swapbytes(typecast(uint8(raw(1:4)), 'uint32'));
        hdr_le = typecast(uint8(raw(1:4)), 'uint32');
        if hdr_be == expectedHeader
            endianness = "big";
        elseif hdr_le == expectedHeader
            endianness = "little";
        end
        detectCount = detectCount + 1;
        if endianness ~= ""
            fprintf("Detected endian = %s\n", endianness);
        end
    end
    if endianness == ""
        continue;
    end

    % 3) 解包 PCM -> reshape -> double
    pcm = typecast(uint8(raw(17:16+payloadBytes)), 'int16');   % int16 vector
    if endianness == "big"
        pcm = swapbytes(pcm);
    end
    pcm = double(reshape(pcm, numChannels, framesPerPacket)).';  % framesPerPacket x Nmic

    % 4) 写入 bufCh（移位写入，避免 concat）
    % bufCh(1:timeWin-writeStep, :) = bufCh(writeStep+1:end, :);
    % bufCh(end-writeStep+1:end, :) = pcm;
    % above version assumes writeStep small; implement as:
    if writeStep < timeWin
        bufCh(1:end-writeStep, :) = bufCh(writeStep+1:end, :);
        bufCh(end-writeStep+1:end, :) = pcm;   % pcm is framesPerPacket x Nmic
    else
        % unlikely since framesPerPacket << timeWin; but safe fallback:
        bufCh = pcm(end-timeWin+1:end, :);
    end

    % 5) 更新瞬时快照用于 Rxx
    % 选择当前包的第一个样本作为瞬时快照（也可使用平均）
    x_inst = pcm(1,:).';   % Nmic x 1
    Rxx = alpha * Rxx + (1 - alpha) * (x_inst * x_inst');   % exponential smoothing

    frameCount = frameCount + 1;

    % 6) 实时时域/FFT 显示（频率降低）
    if toc(fftTimer) > fftUpdateInterval
        fftTimer = tic;

        % 单通道时域显示（displayChannel）
        chanSig = bufCh(:, displayChannel);
        set(hTime, 'YData', chanSig);

        % 单通道 FFT
        win = hann(timeWin);
        Xchan = fft(chanSig .* win, timeWin);
        Pchan = abs(Xchan(1:nFreq));
        set(hFFT, 'YData', 20*log10(Pchan + 1e-12));

        % ========== MVDR 频域处理 ==========
        % 对每通道做 FFT（时间窗上）
        % Xf: Nmic x nFreq
        W = repmat(win, 1, Nmic);
        bufWin = bufCh .* W;            % timeWin x Nmic
        Xall = fft(bufWin, timeWin);    % timeWin x Nmic
        Xall = Xall(1:nFreq, :);        % nFreq x Nmic
        Xall = Xall.';                  % Nmic x nFreq

        % 对每频点计算 MVDR 输出 Y(k) = w_k' * Xall(:,k)
        Y = zeros(nFreq,1);
        for k = 1:nFreq
            d = steerMat(:,k);
            % regularize Rxx
            Rreg = Rxx + diagLoading * trace(Rxx)/Nmic * eye(Nmic);
            % solve R \ d efficiently (backslash)
            rd = Rreg \ d;
            denom = (d' * rd);
            if abs(denom) < 1e-12
                w = zeros(Nmic,1);
            else
                w = rd / denom;
            end
            Y(k) = w' * Xall(:,k);
        end

        mvdr_spec_col = 20*log10(abs(Y) + 1e-12);
        % append to spectrogram buffer
        mvdr_spec_data(:, end+1) = mvdr_spec_col;
        if size(mvdr_spec_data,2) > maxSpecCols
            mvdr_spec_data = mvdr_spec_data(:, end-maxSpecCols+1:end);
        end

        % update MVDR spectrum plot (current column)
        set(hMVDR, 'YData', mvdr_spec_col);

        % update spectrogram plot (CData)
        if enableSpectrogram
            % pad if fewer cols than maxSpecCols for stable display size
            cols = size(mvdr_spec_data,2);
            if cols < maxSpecCols
                padC = zeros(nFreq, maxSpecCols - cols);
                C = [padC mvdr_spec_data];
            else
                C = mvdr_spec_data;
            end
            set(hSpec, 'CData', C);
            drawnow limitrate nocallbacks;
        end

        % 7) 定期保存频谱图为 PNG
        % if toc(lastSaveTime) >= saveInterval
        %     lastSaveTime = tic;
        %     timestamp = datestr(now,'yyyy-mm-dd_HH-MM-SS');
        %     fname = fullfile(outputDir, "mvdr_spec_" + timestamp + ".png");
        %     create figure for saving
        %     fig = figure('Visible','off','Position',[100 100 800 600]);
        %     imagesc((1:size(mvdr_spec_data,2)), freqs, mvdr_spec_data);
        %     axis xy; xlabel('Frame'); ylabel('Frequency (Hz)');
        %     title("MVDR Spectrogram - " + timestamp);
        %     colorbar;
        %     colormap jet;
        %     saveas(fig, fname);
        %     close(fig);
        %     fprintf("Saved %s\n", fname);
        % end

        % ===== 7) 定期保存频谱图为 PNG =====
        if toc(lastSaveTime) >= saveInterval
            lastSaveTime = tic;
            timestamp = datestr(now,'yyyy-mm-dd_HH-MM-SS');
            fname = fullfile(outputDir, "mvdr_spec_" + timestamp + ".png");
        
            % --- 限制频率到 0–8 kHz ---
            idx_8k = freqs <= 8000;        % 频率掩码
            freqs_cut = freqs(idx_8k);     % 限制后的频率轴
            spec_cut  = mvdr_spec_data(idx_8k, :);  % 提取前 8kHz 的频谱
        
            % --- 创建隐藏 figure 保存 ---
            fig = figure('Visible','off','Position',[100 100 800 600]);
            imagesc(1:size(spec_cut,2), freqs_cut, spec_cut);
            axis xy; xlabel('Frame'); ylabel('Frequency (Hz)');
            title("MVDR Spectrogram - " + timestamp);
            colorbar;
            colormap jet;
        
            saveas(fig, fname);
            close(fig);
            fprintf("Saved %s (limited to 0–8 kHz)\n", fname);
        end

    end % fftUpdateInterval

end % while ishandle

% 退出时保存最后一张（如果有）
if ~isempty(mvdr_spec_data)
    timestamp = datestr(now,'yyyy-mm-dd_HH-MM-SS');
    fname = fullfile(outputDir, "mvdr_spec_final_" + timestamp + ".png");
    fig = figure('Visible','off','Position',[100 100 800 600]);
    imagesc((1:size(mvdr_spec_data,2)), freqs, mvdr_spec_data);
    axis xy; xlabel('Frame'); ylabel('Frequency (Hz)');
    title("MVDR Spectrogram - final");
    colorbar; colormap jet;
    saveas(fig, fname);
    close(fig);
    fprintf("Saved final spectrogram %s\n", fname);
end

fprintf("Stopped. Exiting function.\n");
end
