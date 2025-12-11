%% 32位，丢包

function udp_128ch_receiver_matlab()
% UDP 128ch Receiver (MATLAB)
%  - 每包 1040 字节 (16 bytes header + 1024 bytes payload)
%  - payload = 1024 bytes = 512 int16 = 4 frames x 128 channels
%  - header expected = 0xEB9055CC
%
% Usage: run this .m file. Close the figure window to stop and save.

%% --------------- 用户可配置参数 ----------------
localIP = "192.168.0.3";    % 本机 IP（可留空 ""）
localPort = 8080;           % 接收端口
expectedHeader = uint32(hex2dec('EB9055CC'));
numChannels = 128;
framesPerPacket = 4;
bytesPerPacket = 1040;      % 固定
payloadBytes = 1024;
pcm_sample_rate = 192000;   % 硬件标称采样率（每通道）
displayChannel = 1;         % 需要实时显示的通道（1..128）
waveWindowSize = 2000;      % 显示窗口长度 (samples)
saveEveryPackets = 2000;    % 每 N 包自动保存一次到磁盘（防止内存占满）
outputMatPrefix = "capture_";% 保存文件前缀


%% --------------- 初始化与检查 ----------------
fprintf("UDP 128ch Receiver (MATLAB)\n");
fprintf("绑定本地端口 %d ...\n", localPort);

% 创建 udpport（IPv4 datagram）
try
    if localIP == ""
        u = udpport("LocalPort", localPort, "Timeout", 0.5);
    else
        u = udpport("LocalHost", localIP, "LocalPort", localPort, "Timeout", 0.5);
    end
catch ME
    error("无法创建 udpport: %s\n", ME.message);
end

% 用于显示
hFig = figure('Name','UDP 128ch - RealTime Display','NumberTitle','off',...
    'Position',[200 200 1000 500]);

hAx = axes('Parent', hFig);
hPlot = plot(hAx, zeros(waveWindowSize,1), 'b', 'LineWidth', 1.2);
grid(hAx, 'on');
title(hAx, sprintf("Channel %d", displayChannel));
xlabel(hAx, 'Sample'); ylabel(hAx, 'Amplitude');
ylim(hAx, [-1 1]);

% info text
hInfo = annotation('textbox',[0.02 0.02 0.4 0.12],'String','Waiting...','FitBoxToText','on');

% buffers & state
dataBuffer = zeros(waveWindowSize,1);
packetCount = 0;
lastAudCnt = -1;
allData = [];   % will store rows of size numChannels (append frames vertically)
stopped = false;
tStart = tic;

fprintf("开始接收（按关闭图窗停止）...\n");

%% --------------- 主循环 ----------------
try
    while ishandle(hFig)
        % 如果没有足够数据，短暂停
        if u.NumBytesAvailable < bytesPerPacket
            pause(0.001);
            continue;
        end

        % 读取完整包
        raw = read(u, bytesPerPacket, "uint8");
        if numel(raw) ~= bytesPerPacket
            warning("收到字节数(%d)不等于期望(%d)，跳过", numel(raw), bytesPerPacket);
            continue;
        end
        packetCount = packetCount + 1;

        % 解析头部
        
        header  = swapbytes(typecast(uint8(raw(1:4)), 'uint32'));
        aud_cnt = swapbytes(typecast(uint8(raw(5:8)), 'uint32'));
        
        zero1   = swapbytes(typecast(uint8(raw(9:12)), 'uint32'));
        zero2   = swapbytes(typecast(uint8(raw(13:16)), 'uint32'));


        % 若 header 不匹配，尝试判断是否 byte-order 或 数据类型不同
        if header ~= expectedHeader
            % 检查是否为 float32 pattern 或 int16 shifted
            % 直接打印提示并继续（不要立刻终止）
            warning("包 %d: header 不匹配: 0x%X (期望 0x%X).", packetCount, header, expectedHeader);
            % 继续解析 payload 以便诊断（不返回）
        end

        % 取音频负载并转为 int16
        aud_data = raw(17 : 16 + payloadBytes);  % 1024 bytes

        pcm_int16 = swapbytes(typecast(uint8(aud_data), 'int16'));


        % reshape 成 [channels × framesPerPacket] = 128 × 4
        if numel(pcm_int16) ~= (numChannels * framesPerPacket)
            warning("包 %d: payload 转 int16 后长度(%d)与期望(%d)不符，跳过。", packetCount, numel(pcm_int16), numChannels*framesPerPacket);
            continue;
        end

        pcmMatrix = reshape(pcm_int16, [numChannels, framesPerPacket]);  % 128 x 4

        % 丢包检测（基于 aud_cnt）
        if lastAudCnt >= 0 && aud_cnt ~= lastAudCnt + 1
            missed = aud_cnt - lastAudCnt - 1;
            if missed > 0
                fprintf("丢包: 上一 aud_cnt=%d, 当前=%d, 丢失 %d 包\n", lastAudCnt, aud_cnt, missed);
            end
        end
        lastAudCnt = aud_cnt;

        % 保存所有通道数据（按行追加，行数 = frames accumulated × channels）
        % 为便于后续处理，我们把每帧（列）按行追加：每次追加 framesPerPacket 行，每行128列
        allData = [allData; double(pcmMatrix.')];  % append frames as rows (frames x channels)

        % 取显示通道的数据（列向量）
        chData = double(pcmMatrix(displayChannel, :)).';  % 4×1

        % 更新滚动 buffer
        L = numel(chData);
        if L <= waveWindowSize
            dataBuffer = [dataBuffer(L+1:end); chData];
        else
            dataBuffer = chData(end-waveWindowSize+1:end);
        end

        % 更新绘图
        set(hPlot, 'YData', dataBuffer);
        ymax = max(abs(dataBuffer));
        if ymax > 0
            ylim(hAx, [-ymax*1.1, ymax*1.1]);
        end

        % 更新文本信息
        infoTxt = sprintf("包# %d | aud_cnt=%d | 总帧数=%d | 运行=%.1fs", ...
                          packetCount, aud_cnt, size(allData,1), toc(tStart));
        set(hInfo, 'String', infoTxt);

        drawnow limitrate nocallbacks;

        % 周期性保存（减小内存峰值）
        if mod(packetCount, saveEveryPackets) == 0
            fname = sprintf("%s%d.mat", outputMatPrefix, packetCount);
            fprintf("周期性保存到 %s (totalFrames=%d)\n", fname, size(allData,1));
            save(fname, 'allData', 'pcm_sample_rate', 'numChannels', 'framesPerPacket', '-v7.3');
        end
    end
catch ME
    fprintf("捕获到异常: %s\n", ME.message);
    disp(ME.stack(1));
end

% --------------- 保存.mat格式 ----------------
fprintf("接收结束，开始最终保存...\n");
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
matname = sprintf('%s%s.mat', outputMatPrefix, timestamp);

% allData: rows = totalFrames, cols = numChannels
if ~isempty(allData)
    save(matname, 'allData', 'pcm_sample_rate', 'numChannels', 'framesPerPacket', '-v7.3');
    fprintf("已保存 MAT: %s (frames=%d, channels=%d)\n", matname, size(allData,1), size(allData,2));

    %保存为多通道 wav（samples x channels）
    wavname = sprintf('%s%s.wav', outputMatPrefix, timestamp);
    %allData rows are frames (time), columns are channels -> directly write
    %归一化到 [-1,1]
    audioMatrix = allData;  % frames x channels
    maxv = max(abs(audioMatrix(:)));
    if maxv > 0
        audioNorm = audioMatrix / double(maxv);
    else
        audioNorm = audioMatrix;
    end

    try
        audiowrite(wavname, audioNorm, pcm_sample_rate);
        fprintf("已保存 WAV: %s\n", wavname);
    catch WME
        warning("保存 WAV 失败: %s", WME.message);
    end
else
    fprintf("无数据保存。\n");
end

%清理 udpport
if exist('u','var')
    clear u;
end

fprintf("退出。\n");
end


% % ---------------- 保存 .mat 文件 -----------------
% fprintf("接收结束，开始最终保存...\n");
% timestamp = datestr(now, 'yyyymmdd_HHMMSS');
% matname = sprintf('%s%s.mat', outputMatPrefix, timestamp);
% 
% if ~isempty(allData)
% 
%     % 保存为 MAT
%     save(matname, 'allData', 'pcm_sample_rate', 'numChannels', 'framesPerPacket', '-v7.3');
%     fprintf("已保存 MAT: %s (frames=%d, channels=%d)\n", ...
%              matname, size(allData,1), size(allData,2));
% 
%     % ---------------- 保存 WAV 文件 ----------------
%     wavname = sprintf('%s%s.wav', outputMatPrefix, timestamp);
% 
%     % allData 可能是 double/int16，需要确保满足 WAV 要求
%     % 如果是 double → 转回 int16 再写入
%     audio_int16 = int16(allData);
% 
%     % 转为 [-1,1] 区间的 double
%     audio_norm = double(audio_int16) / 32768;
% 
%     try
%         audiowrite(wavname, audio_norm, pcm_sample_rate);
%         fprintf("已保存 WAV: %s\n", wavname);
%     catch WME
%         warning("保存 WAV 失败: %s", WME.message);
%     end
% 
% else
%     fprintf("无数据保存。\n");
% end
% 
% if exist('u','var')
%     clear u;
% end
% fprintf("退出。\n");
% end



