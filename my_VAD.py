# 我的VAD
import math
import librosa
import numpy as np
import matplotlib.pyplot as plt

def cal_per_Energy(wave_data) :
    #计算每帧的短时能量与rms,以256个采样点为一帧
    energy = []
    sum = 0
    rms1 = []
    for i in range(len(wave_data)) :
        sum = sum + (wave_data[i] * wave_data[i])
        if (i+1) % 256 == 0:
            energy.append(sum)
            if sum != 0:
                rms = 10 * math.log10(sum/256)
                rms1.append(rms)
            else:
                rms = 10 * math.log10(0.001/256)
                rms1.append(rms)
            sum = 0
        elif i == len(wave_data) - 1:
            energy.append(sum)
            if sum != 0:
                rms = 10 * math.log10(sum/256)
                rms1.append(rms)
            else:
                rms = 10 * math.log10(0.001/256)
                rms1.append(rms)
    return energy,rms1

def cal_silent(rms_list):
    #计算无声段是从哪一帧开始的
    for i in range(len(rms_list)):
        if (str(all([rms_db <= -58 for rms_db in rms_list[i:]])) == 'True'):
            silent_index = i
            break
        else:
            silent_index = 'no silent segment'  #表示没有无声段
    return silent_index

def duration_time(y,sample_rate):
    #计算音频持续时间
    duration = librosa.get_duration(y,sr=sample_rate)
    return duration

def cal_rms(y):
    #计算每帧的rms,以1024个采样点为一帧
    sum = 0
    rms1 = []
    for i in range(len(y)) :
        sum = sum + (y[i] * y[i])
        if (i+1) % 1024 == 0:
            if sum != 0:
                rms = 10 * math.log10(sum/1024)
                rms1.append(rms)
            else:
                rms = 10 * math.log10(0.0001/1024)
                rms1.append(rms)
            sum = 0
        elif i == len(y) - 1:
            if sum != 0:
                rms = 10 * math.log10(sum/1024)
                rms1.append(rms)
            else:
                rms = 10 * math.log10(0.0001/1024)
                rms1.append(rms)
    return rms1

def cal_possible1(y,delta_data):
    #判断是不是全是噪声
    threshold = 0.1     #噪声平稳性在0.1之内的,都判定为噪声
    rms_threshld = 18   #或者最大rms与最小rms的差值为20之内,都判定为噪声
    rms = cal_rms(y)
    delta_rms = max(rms) - min(rms)
    noise_all = 'not_all_noise'
    #两者满足其中一个条件的,就判定为噪声
    if str(all(delta < threshold for delta in delta_data)) == 'True'  or delta_rms < rms_threshld:
        noise_all = 'all_noise'
    return noise_all

def cal_possible(delta_data):
    #一段一段检测到底是噪声还是回声,目前只能区分平稳噪声和语音,但是不能区分非平稳噪声
    threshold = 0.17
    if str(all(delta < threshold for delta in delta_data)) == 'True':
        echo_or_noise = 'noise'
    else :
        echo_or_noise = 'echo'
    return echo_or_noise


def cal_zcr(y,frame_length,hop_length):
    #求过零率,并根据过零率设计算法区分噪声和回声
    zcr = librosa.feature.zero_crossing_rate(y,frame_length=frame_length,hop_length=hop_length) #默认帧长是2048,帧移是512
    zcr = zcr[0]
    zcr_group = []
    # save_path = '/Users/gesimeng/Desktop/5-13test/第三轮数据/oppor17/2.9.0/zcr/'+case+'.jpg'
    # plot_figure1(zcr,save_path)

    #将相邻16帧分为一组,每次向右移动一帧
    for part in range(0,len(zcr)-15,1):
        group = zcr[part:part+16]
        zcr_group.append(group)

    #求出每一组中最大值与最小值的差
    delta_list = []
    for j in range(0,len(zcr_group)):  #这里故意丢掉几组,因为从有噪到无噪,过零率会发生突变,丢掉两组应该不影响结果
        group_max = max(zcr_group[j])
        group_min = min(zcr_group[j])
        delta = group_max - group_min
        delta_list.append(delta)
    return delta_list

def ave_energy(wave_data):
    #计算所有帧的平均能量
    total_energy = 0
    for i in range(len(wave_data)):
        total_energy += (wave_data[i] * wave_data[i])
    aveEnergy = total_energy/(len(wave_data)/256)
    return aveEnergy

def continusFind(num_list):
    '''
    列表中连续数字段寻找
    '''
    num_list.sort()
    s = 1
    find_list = []
    have_list = []
    while s <= len(num_list)-1:
        if (num_list[s]-num_list[s-1]) == 1:
            flag = s-1
            while (s <= (len(num_list)-1)) and (num_list[s]-num_list[s-1]==1):
                s += 1
            find_list.append(num_list[flag:s])
            have_list += num_list[flag:s]
        else:
            s += 1
    return find_list

def fill_consecutive(index_list):
    #把连续数补齐,也就是对大于平均能量的帧,都连续起来,两个之间的差值是30帧,也就是13*256/16000=0.208s
    k = 0
    while k<len(index_list)-1:
        if (index_list[k+1][0]-index_list[k][-1]) < 30:
            index_list[k] = [i for i in range(index_list[k][0],index_list[k+1][-1]+1)]
            index_list.remove(index_list[k+1])
        else:
            k += 1
    return index_list

sample_rate = 16000
frame_length = 512
hop_length = 128

y,sr = librosa.load('test_data/13.wav',sr=sample_rate)
y_last = y
_,rms = cal_per_Energy(y)
time = duration_time(y,sample_rate)

thre = np.zeros([1,len(y)])

silent_index = cal_silent(rms)
if silent_index == 'no silent segment':  # 表示没有无声段
    silent_time = time
    y = y
else:
    silent_time = (silent_index * time) / len(rms)  # 无声段开始的时刻
    y = y[0:int(silent_time * sample_rate)]

if (len(y) != 0):  # 说明不是一开始就进入无声段
    time_voice = duration_time(y, sample_rate)

    # 这里可以先来一个噪声检测,就是检测是不是全是噪声,如果是,那就返回有噪声,如果不是,那就去计算.
    delta_list1 = cal_zcr(y, frame_length=frame_length, hop_length=hop_length)
    all_noise = cal_possible1(y, delta_list1)
    if all_noise == 'all_noise':
        thre = thre
    else:
        energy, rms1 = cal_per_Energy(y)
        aveEnergy = ave_energy(y)
        index_list = []
        noise_list = []

        for i in range(len(energy)):
                if (energy[i] >= aveEnergy) and (rms1[i] > -58):
                    index_list.append(i)
                if energy[i] < aveEnergy:  # 把噪音的点记下来
                    noise_list.append(i)
        index_list_continue = continusFind(index_list)

        # 大于平均能量值的帧
        frame_list = fill_consecutive(index_list_continue)

        y_list = []
        start = []
        end = []
        for j in range(len(frame_list)):
                start_index = frame_list[j][0] * 256  # 语音开始的起点帧数
                end_index = (frame_list[j][-1] + 1) * 256  # 语音结束的终点帧数
                start.append(start_index)
                end.append(end_index)
                y_list.append(y[start_index:end_index])  # 一段一段的进行检测

        # echo_dur_list = []
        # 到目前为止检测到了大于平均能量的点,也就是语音或者噪声
        # 下一步,再看看是噪声还是语音,根据过零率
        echo_noise_list = []
        for l in range(len(y_list)):
                delta_list = cal_zcr(y_list[l], frame_length, hop_length)
                echo_or_noise = cal_possible(delta_list)
                echo_noise_list.append(echo_or_noise)

        # 现在将平稳噪声和回声区分开了,直接求最后一个回声的end_time就是收敛时间
        if str(all(echo_noise == 'noise' for echo_noise in echo_noise_list)) == 'True':
            thre = thre

        elif str(all(echo_noise == 'echo' for echo_noise in echo_noise_list)) == 'True':
            for i in range(len(start)):
                thre[0][start[i]:end[i]] = [1] * (end[i] - start[i])

        else:

                for index in range(len(echo_noise_list)):
                    if echo_noise_list[index] == 'echo':
                        thre[0][start[index]:end[index]] = [1] * (end[index] - start[index])
                        start_time = (start[index] * time_voice) / len(y)
                        end_time = (end[index] * time_voice) / len(y)
                        echo_dur_time = end_time - start_time

else:
    thre = thre

plt.subplot(2, 1, 1)
plt.plot(y_last)
plt.title('Time Signal')
plt.ylim([-1,1])

plt.subplot(2, 1, 2)
plt.plot(thre[0])
plt.xlabel('frame')
plt.ylabel('Prob')

plt.tight_layout()
plt.show()
