function udp_128ch_receiver()

%% ---------------- 用户参数 ----------------
localIP = "192.168.0.3";
localPort = 8080;

expectedHeader = uint32(hex2dec('EB9055CC'));
numChannels = 128;
framesPerPacket = 4;
bytesPerPacket  = 1040;
payloadBytes    = 1024;

Fs = 192000;
displayChannel = 1;  % 显示第一个通道

timeWin = 4096;      % 显示窗口大小
fftUpdateInterval = 0.05;

% 声谱图参数
spectrogramTimeLength = 2;  % 声谱图显示的时间长度（秒）
spectrogramOverlap = 0.8;   % 重叠比例
spectrogramNFFT = 1024;     % FFT点数

% 时间波形图参数
timeDomainLength = 0.05;    % 时间波形图显示的长度（秒）

rawTempFile = "udp_capture_tmp.raw";

%% ---------------- 创建 UDP 口 ----------------
if localIP == ""
    u = udpport("LocalPort", localPort, "Timeout", 0.5);
else
    u = udpport("LocalHost", localIP, "LocalPort", localPort, "Timeout", 0.5);
end

fprintf("UDP Receiver started at %s:%d \n", localIP, localPort);
fprintf("Displaying Channel %d data\n", displayChannel);
fprintf("Sampling rate: %d Hz\n", Fs);

%% ---------------- 创建界面 ----------------
hFig = figure('Name','UDP 128ch - realtime','NumberTitle','off', 'Position', [100 100 1200 1000]);

% 第一行：两个时域图
% 时域图（样本索引）
subplot(3,2,1);
hTime = plot(zeros(timeWin,1)); 
grid on; axis tight;
title(sprintf("Channel %d - Time Domain (Sample Index)", displayChannel));
xlabel("Sample Index"); ylabel("Amplitude");
ylim([-35000 35000]);

% 时域图（时间轴）
subplot(3,2,2);
timeAxis = (0:timeWin-1)/Fs;
hTimeDomain = plot(timeAxis, zeros(timeWin,1));
grid on; axis tight;
title(sprintf("Channel %d - Time Domain (Seconds)", displayChannel));
xlabel("Time (s)"); ylabel("Amplitude");
ylim([-35000 35000]);

% 第二行：频域图
subplot(3,2,[3,4]);
hFFT = plot(zeros(timeWin/2,1)); 
grid on; axis tight;
title(sprintf("Channel %d - Frequency Spectrum", displayChannel));
xlabel("Frequency (Hz)"); ylabel("Magnitude (dB)");
xlim([0 10000]);

% 第三行：声谱图
subplot(3,2,[5,6]);
% 预初始化声谱图
initialSpectrogramData = zeros(spectrogramNFFT/2, 100);
hSpectrogram = imagesc([0, spectrogramTimeLength], [0, Fs/2], initialSpectrogramData);
title(sprintf("Channel %d - Spectrogram", displayChannel));
xlabel("Time (s)"); ylabel("Frequency (Hz)");
colorbar;
axis xy;  % 确保低频在底部
caxis([-80 0]);  % 设置颜色范围
ylim([0 10000]); % 限制频率范围以便观察1kHz信号

drawnow;

%% ---------------- 状态变量 ----------------
fid = fopen(rawTempFile,"w");
buf = zeros(timeWin,1);
fftTimer = tic;

% 声谱图相关变量
spectrogramBuffer = [];
spectrogramTimer = tic;
spectrogramUpdateInterval = 0.2;  % 声谱图更新间隔
spectrogramSamplesNeeded = round(spectrogramTimeLength * Fs);

% 时间波形图相关变量
timeDomainBuffer = [];
timeDomainSamplesNeeded = round(timeDomainLength * Fs);

endianness = "";
detectCount = 0;

packetCount = 0;
dataReceived = 0;
fprintf("Waiting for data...\n");

%% ---------------- 主循环 ----------------
while ishandle(hFig)
    raw = read(u, bytesPerPacket, "uint8");
    if numel(raw) ~= bytesPerPacket
        pause(0.001);
        continue;
    end

    packetCount = packetCount + 1;
    dataReceived = dataReceived + 1;
    
    % 检测字节序
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

    % 提取PCM数据
    pcm = typecast(uint8(raw(17:16+payloadBytes)), 'int16');

    if endianness == "big"
        pcm = swapbytes(pcm);
    end

    pcm = reshape(pcm, [numChannels, framesPerPacket]);

    % 保存到临时文件
    fwrite(fid, pcm.', 'int16');

    % 提取第一个通道的数据
    samples = double(pcm(displayChannel,:)).';
    L = numel(samples);
    buf = [buf(L+1:end); samples];
    
    % 更新时间波形图缓冲区
    timeDomainBuffer = [timeDomainBuffer; samples];
    if length(timeDomainBuffer) > timeDomainSamplesNeeded
        timeDomainBuffer = timeDomainBuffer(end-timeDomainSamplesNeeded+1:end);
    end
    
    % 更新声谱图缓冲区
    spectrogramBuffer = [spectrogramBuffer; samples];
    if length(spectrogramBuffer) > spectrogramSamplesNeeded
        spectrogramBuffer = spectrogramBuffer(end-spectrogramSamplesNeeded+1:end);
    end
    
    % 更新时域图（样本索引）
    set(hTime,'YData',buf);
    
    % 更新时域图（时间轴）
    if length(timeDomainBuffer) >= timeWin
        % 使用最新的timeWin个样本
        currentData = timeDomainBuffer(end-timeWin+1:end);
        timeAxis = (0:length(currentData)-1)/Fs;
        set(hTimeDomain,'XData',timeAxis,'YData',currentData);
    elseif length(timeDomainBuffer) > 0
        % 如果数据不足，使用所有可用数据
        currentData = timeDomainBuffer;
        timeAxis = (0:length(currentData)-1)/Fs;
        set(hTimeDomain,'XData',timeAxis,'YData',currentData);
    end
    
    % 定期更新频域图
    if toc(fftTimer) > fftUpdateInterval
        fftTimer = tic;
        NFFT = timeWin;
        
        % 应用窗函数并进行FFT
        windowedData = buf .* hann(timeWin);
        Y = fft(windowedData, NFFT);
        P = abs(Y(1:NFFT/2));
        
        % 转换为dB
        magnitude_dB = 20*log10(P + 1e-6);
        
        % 创建频率轴
        freqAxis = (0:(NFFT/2-1)) * (Fs/NFFT);
        
        % 更新频域图
        set(hFFT,'XData',freqAxis,'YData',magnitude_dB);
        
        % 在标题中显示包计数
        set(hFig, 'Name', sprintf('UDP 128ch - realtime (Packets: %d, Data: %d)', packetCount, dataReceived));
    end
    
    % 定期更新声谱图
    if toc(spectrogramTimer) > spectrogramUpdateInterval && length(spectrogramBuffer) >= spectrogramNFFT
        spectrogramTimer = tic;
        
        try
            % 计算声谱图
            [S, F, T] = spectrogram(spectrogramBuffer, ...
                                   hann(spectrogramNFFT), ...
                                   round(spectrogramNFFT * spectrogramOverlap), ...
                                   spectrogramNFFT, Fs);
            
            % 转换为dB
            S_dB = 20*log10(abs(S) + 1e-6);
            
            % 更新声谱图显示
            set(hSpectrogram, 'XData', [0, max(T)], 'YData', [0, max(F)], 'CData', S_dB);
            
            % 更新标题显示声谱图信息
            subplot(3,2,[5,6]);
            title(sprintf("Channel %d - Spectrogram (Size: %dx%d)", displayChannel, size(S_dB,1), size(S_dB,2)));
            
            fprintf('Spectrogram updated: %dx%d, Freq range: [0, %.1f] Hz, Time range: [0, %.2f] s\n', ...
                    size(S_dB,1), size(S_dB,2), max(F), max(T));
        catch ME
            fprintf('Error updating spectrogram: %s\n', ME.message);
        end
    end

    drawnow limitrate nocallbacks;

end

%% ---------------- 清理并写 WAV ----------------
fclose(fid);

fprintf("\n窗口关闭，开始生成 WAV 文件...\n");

fid = fopen(rawTempFile,"r");
rawAll = fread(fid, 'int16');
fclose(fid);

delete(rawTempFile);

rawAll = reshape(rawAll, [], numChannels);

%% ===== 保存第一个通道的 WAV 文件 =====
wavData = rawAll(:, displayChannel) / 32768;
audiowrite("udp128_record_ch1.wav", wavData, Fs);

fprintf("保存完成: udp128_record_ch%d.wav (第一个通道)\n", displayChannel);
fprintf("总共接收数据包: %d, 有效数据包: %d\n", packetCount, dataReceived);

end