function udp_128ch_receiver()

%% ---------------- 用户参数 ----------------
localIP = "192.168.0.3";
localPort = 8080;

expectedHeader = uint32(hex2dec('EB9055CC'));
numChannels = 128;
framesPerPacket = 4;             % 每包 4 帧
bytesPerPacket  = 1040;
payloadBytes    = 1024;

Fs = 192000;                     % 采样率
displayChannel = 1;

timeWin = 4096;                  % 显示窗口大小
fftUpdateInterval = 0.05;        % 每 50ms 计算一次 FFT

rawTempFile = "udp_capture_tmp.raw"; % 临时文件（退出时转 WAV）

%% ---------------- 创建 UDP 口 ----------------
if localIP == ""
    u = udpport("LocalPort", localPort, "Timeout", 0.5);
else
    u = udpport("LocalHost", localIP, "LocalPort", localPort, "Timeout", 0.5);
end

fprintf("UDP Receiver started at %s:%d \n", localIP, localPort);

%% ---------------- 创建界面 ----------------
hFig = figure('Name','UDP 128ch - realtime','NumberTitle','off');
subplot(2,1,1);
hTime = plot(zeros(timeWin,1)); grid on;
title(sprintf("Channel %d (time domain)",displayChannel));
xlabel("Sample"); ylabel("Amplitude");

subplot(2,1,2);
hFFT = plot(zeros(timeWin/2,1)); grid on;
title("Magnitude Spectrum");
xlabel("Frequency (Hz)"); ylabel("Magnitude (dB)");

drawnow;

%% ---------------- 状态变量 ----------------
fid = fopen(rawTempFile,"w");     % 低内存连续写入
buf = zeros(timeWin,1);
fftTimer = tic;

endianness = "";                  % "big" or "little"
detectCount = 0;

fprintf("Waiting for data...\n");

%% ---------------- 主循环 ----------------
while ishandle(hFig)

    % 读一个包
    raw = read(u, bytesPerPacket, "uint8");
    if numel(raw) ~= bytesPerPacket
        pause(0.001);
        continue;
    end

    %% ===== 自动判断端序 =====
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

    %% 若还未确定端序，继续跳过
    if endianness == ""
        continue;
    end

    %% ===== 解析 payload 1024 字节 → 512 int16 =====
    pcm = typecast(uint8(raw(17:16+payloadBytes)), 'int16');

    if endianness == "big"
        pcm = swapbytes(pcm);
    end

    % reshape 为 128 x 4
    pcm = reshape(pcm, [numChannels, framesPerPacket]);

    %% ===== 写入原始数据（低内存）=====
    fwrite(fid, pcm.', 'int16');   % 写入 4x128

    %% ===== 更新时域波形 =====
    samples = double(pcm(displayChannel,:)).';
    L = numel(samples);
    buf = [buf(L+1:end); samples];
    set(hTime,'YData',buf);

    %% ===== 周期性更新 FFT =====
    if toc(fftTimer) > fftUpdateInterval
        fftTimer = tic;
        NFFT = timeWin;
        Y = fft(buf .* hann(timeWin), NFFT);
        P = abs(Y(1:NFFT/2));

        freqAxis = (0:(NFFT/2-1)) * (Fs/NFFT);
        set(hFFT,'XData',freqAxis,'YData',20*log10(P+1e-6));
    end

    drawnow limitrate nocallbacks;

end

%% ---------------- 清理并写 WAV ----------------
fclose(fid);

fprintf("\n窗口关闭，开始生成 WAV 文件...\n");

fid = fopen(rawTempFile,"r");
rawAll = fread(fid, 'int16');
fclose(fid);

delete(rawTempFile); % 删除临时文件

% reshape 成 [总帧, 128]
rawAll = reshape(rawAll, [], numChannels);

audiowrite("udp128_record.wav", rawAll / 32768, Fs);

fprintf("保存完成: udp128_record.wav\n");




