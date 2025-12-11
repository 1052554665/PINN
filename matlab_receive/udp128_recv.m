%% UDP PCM波形实时显示 - 完整版本
clear; clc;

localPort = 8080;   % 端口号改为接收端端口，接收端口号 (0x8080 = 32960)
udpPacketSize = 1040;   % % 每包字节数: 16字节头 + 1024字节音频数据
numChannels = 128;         % 通道数
pcm_sample_rate = 192000;   % 采集板采样率
waveWindowSize = 500;  % 显示2000个采样点

fprintf('=== PCM波形实时显示 ===\n');
fprintf('监听端口: %d\n', localPort);

% 创建UDP连接
try
    u = udpport("LocalPort", localPort);
    read(u, 100, "uint8")   % 如果卡住不返回，说明采集板还没发包。如果立刻返回一串数字，说明有数据到达。
    fprintf('✅ 成功绑定端口：%d\n', localPort);
catch err
    error('❌ 端口绑定失败：%s\n', err.message);
end

% 创建图形窗口
fig = figure('Name', 'PCM波形实时显示', 'NumberTitle', 'off', ...
    'Position', [100, 100, 1200, 600]);

% 主波形显示
subplot(2,1,1);
hPlot = plot(zeros(waveWindowSize,1), 'b-', 'LineWidth', 1.2);
title('实时PCM波形', 'FontSize', 14);
xlabel('采样点', 'FontSize', 12);
ylabel('幅值', 'FontSize', 12);
grid on;

% 信息显示区域
subplot(2,1,2);
hInfo = text(0.02, 0.8, '等待数据...', 'FontSize', 11, 'VerticalAlignment', 'top');
axis([0 1 0 1]);
axis off;

% 数据缓冲区初始化
dataBuffer = zeros(waveWindowSize, 1);
packetCount = 0;
isRunning = true;

fprintf('等待数据中...\n');

try
    while isRunning && ishandle(fig)
        if u.NumBytesAvailable >= udpPacketSize
            % 读取UDP数据
            udpData = read(u, udpPacketSize, "uint8");
            packetCount = packetCount + 1;
            
            % 解析为16位PCM数据
            if mod(length(udpData), 2) == 0
                header = typecast(udpData(1:4), 'uint32');
                aud_cnt = typecast(udpData(5:8), 'uint32');
                aud_data = udpData(17:end);     % 去掉前16字节
                pcmData = typecast(aud_data, 'int16');  % 512个采样点
                
                % 更新数据缓冲区
                newSamples = length(pcmData);
                if newSamples <= waveWindowSize
                    dataBuffer = [dataBuffer(newSamples+1:end); double(pcmData(:))];
                else
                    dataBuffer = double(pcmData(end-waveWindowSize+1:end));
                end
                
                % 更新波形显示
                set(hPlot, 'YData', dataBuffer);
                
                % 自动调整Y轴范围
                currentMax = max(abs(dataBuffer));
                if currentMax > 100  % 只有有实际数据时才调整范围
                    ylim([-currentMax*1.1, currentMax*1.1]);
                end
                
                % 更新信息显示
                infoStr = sprintf(['数据包: %d\n', ...
                                  '数据长度: %d 字节 (%d 样本)\n', ...
                                  '幅值范围: %d 到 %d\n', ...
                                  '最新值: %d\n', ...
                                  '时间: %s'], ...
                    packetCount, length(udpData), newSamples, ...
                    min(pcmData), max(pcmData), pcmData(end), datestr(now));
                set(hInfo, 'String', infoStr);
                
                % 显示前几个包的详细信息
                if packetCount <= 3
                    fprintf('数据包 #%d: %d字节, 幅值范围: %d 到 %d\n', ...
                        packetCount, length(udpData), min(pcmData), max(pcmData));
                end
                
                drawnow;
            end
            
            % 可选：限制处理的数据包数量用于测试
            % if packetCount >= 1000
            %     fprintf('已处理1000个数据包，停止接收\n');
            %     break;
            % end
            
        else
            pause(0.001);  % 降低CPU占用
        end
    end
catch ME
    fprintf('程序异常: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('异常位置: %s (第%d行)\n', ME.stack(1).name, ME.stack(1).line);
    end
end

% 清理资源
if exist('u', 'var')
    clear u;
end

fprintf('程序结束，共接收 %d 个数据包\n', packetCount);