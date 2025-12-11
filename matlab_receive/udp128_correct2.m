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
displayChannel = 1;

timeWin = 4096;
fftUpdateInterval = 0.05;

rawTempFile = "udp_capture_tmp.raw";

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
fid = fopen(rawTempFile,"w");
buf = zeros(timeWin,1);
fftTimer = tic;

endianness = "";
detectCount = 0;

fprintf("Waiting for data...\n");

%% ---------------- 主循环 ----------------
while ishandle(hFig)
    raw = read(u, bytesPerPacket, "uint8");
    if numel(raw) ~= bytesPerPacket
        pause(0.001);
        continue;
    end

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

    pcm = typecast(uint8(raw(17:16+payloadBytes)), 'int16');

    if endianness == "big"
        pcm = swapbytes(pcm);
    end

    pcm = reshape(pcm, [numChannels, framesPerPacket]);

    fwrite(fid, pcm.', 'int16');

    samples = double(pcm(displayChannel,:)).';
    L = numel(samples);
    buf = [buf(L+1:end); samples];
    set(hTime,'YData',buf);

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

delete(rawTempFile);

rawAll = reshape(rawAll, [], numChannels);

%% ===== 保存单通道 WAV =====
wavData = rawAll(:, displayChannel) / 32768;
audiowrite("udp128_record_ch1.wav", wavData, Fs);

fprintf("保存完成: udp128_record_ch%d.wav\n", displayChannel);
