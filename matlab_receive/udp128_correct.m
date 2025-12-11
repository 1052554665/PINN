%% correct version

function udp_128ch_receiver_matlab()
% UDP 128ch Receiver (MATLAB)
%  - 每包 1040 字节 (16 bytes header + 1024 bytes payload)
%  - payload = 1024 bytes = 512 int16 = 4 frames x 128 channels
%  - 数据均为大端序（big-endian）
%
% Usage: run this .m file. 关闭图窗停止并自动保存

%% ---------------- 用户参数 ----------------
localIP = "192.168.0.3";
localPort = 8080;
expectedHeader = uint32(hex2dec('EB9055CC'));

numChannels = 128;
framesPerPacket = 4;
bytesPerPacket = 1040;
payloadBytes    = 1024;

pcm_sample_rate = 192000;
displayChannel  = 1;
waveWindowSize  = 2000;
saveEveryPackets = 2000;
outputMatPrefix = "capture_";

%% ---------------- 创建 UDP 口 ----------------
fprintf("UDP Receiver 启动，端口 %d\n", localPort);

if localIP == ""
    u = udpport("LocalPort", localPort, "Timeout", 0.5);
else
    u = udpport("LocalHost", localIP, "LocalPort", localPort, "Timeout", 0.5);
end

%% ---------------- 创建绘图窗口 ----------------
hFig = figure('Name','UDP 128ch - RealTime Display','NumberTitle','off',...
    'Position',[200 200 1000 500]);

hAx = axes('Parent',hFig);
hPlot = plot(hAx, zeros(waveWindowSize,1),'b','LineWidth',1.2);
grid on;
title(hAx, sprintf("Channel %d", displayChannel));
xlabel('Sample'); ylabel('Amplitude');
ylim([-1 1]);

hInfo = annotation('textbox',[0.02 0.02 0.4 0.12],'String','Waiting...');

%% 状态变量
dataBuffer = zeros(waveWindowSize,1);
allData = [];
packetCount = 0;
lastAudCnt = -1;
tStart = tic;

fprintf("开始接收...\n");

%% ---------------- 主循环 ----------------
try
    while ishandle(hFig)

        if u.NumBytesAvailable < bytesPerPacket
            pause(0.001);
            continue;
        end

        raw = read(u, bytesPerPacket, "uint8");
        if numel(raw) ~= bytesPerPacket
            warning("收到字节数不匹配，丢弃");
            continue;
        end
        packetCount = packetCount + 1;

        %% ===== 解析 16 字节头部（大端序）=====
        header  = typecast(uint8(raw(1:4)) , 'uint32');  header  = swapbytes(header);
        aud_cnt = typecast(uint8(raw(5:8)) , 'uint32');  aud_cnt = swapbytes(aud_cnt);
        zero1   = typecast(uint8(raw(9:12)), 'uint32');  zero1   = swapbytes(zero1);
        zero2   = typecast(uint8(raw(13:16)),'uint32');  zero2   = swapbytes(zero2);

        if header ~= expectedHeader
            warning("包 %d：Header 不匹配: 0x%X", packetCount, header);
        end

        %% ===== 解析 Payload: 1024 字节 → 512 个 int16（大端序）======
        aud_bytes = raw(17:16+payloadBytes);
        pcm_int16 = typecast(uint8(aud_bytes), 'int16');
        pcm_int16 = swapbytes(pcm_int16);     % 大端序

        if numel(pcm_int16) ~= numChannels*framesPerPacket
            warning("payload 长度错误，丢弃");
            continue;
        end

        pcmMatrix = reshape(pcm_int16, [numChannels, framesPerPacket]);

        %% ===== 丢包检测（24bit aud_cnt）=====
        aud_cnt24 = bitand(aud_cnt, uint32(hex2dec('FFFFFF')));

        if lastAudCnt >= 0
            diff24 = mod(double(aud_cnt24) - double(lastAudCnt), 2^24);
            if diff24 ~= 1
                fprintf("丢包：上一=%u 当前=%u，丢失 %u 包\n", ...
                    lastAudCnt, aud_cnt24, diff24-1);
            end
        end
        lastAudCnt = aud_cnt24;

        %% ===== 存储数据 (每包追加 4 行 x 128 列) =====
        allData = [allData; double(pcmMatrix.')];

        %% ===== 更新实时曲线 =====
        chData = double(pcmMatrix(displayChannel, :)).';

        L = numel(chData);
        dataBuffer = [dataBuffer(L+1:end); chData];

        set(hPlot,'YData',dataBuffer);
        ymax = max(abs(dataBuffer));
        if ymax > 0
            ylim(hAx, [-ymax*1.1 ymax*1.1]);
        end

        set(hInfo,'String', sprintf("包# %d | aud_cnt=%u | 总帧=%u | 运行 %.1fs", ...
            packetCount, aud_cnt24, size(allData,1), toc(tStart)));
        drawnow limitrate nocallbacks;

        %% 周期性自动保存
        if mod(packetCount, saveEveryPackets)==0
            fname = sprintf("%s%d.mat", outputMatPrefix, packetCount);
            fprintf("自动保存 %s\n", fname);
            save(fname, 'allData','pcm_sample_rate','numChannels','framesPerPacket','-v7.3');
        end
    end

catch ME
    fprintf("异常: %s\n", ME.message);
end
