#谱熵法进行端点检测
import  numpy as  np
import matplotlib.pyplot as plt
from soundBase import *
import os

def enframe(x, win, inc=None):
    #分帧
    nx = len(x)
    if isinstance(win, list) or isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # 设置为帧长
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    #print(frameout)
    return frameout

def FrameTimeC(frameNum, frameLen, inc, fs):
    #求每一帧开始的时刻
    ll = np.array([i for i in range(frameNum)])
    return ((ll - 1) * inc + frameLen / 2) / fs

def findSegment(express):
    """
    分割成語音段
    :param express:
    :return:
    """
    if express[0] == 0:
        voiceIndex = np.where(express)
    else:
        voiceIndex = express
    d_voice = np.where(np.diff(voiceIndex) > 1)[0]
    voiceseg = {}
    if len(d_voice) > 0:
        for i in range(len(d_voice) + 1):
            seg = {}
            if i == 0:
                st = voiceIndex[0]
                en = voiceIndex[d_voice[i]]
            elif i == len(d_voice):
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[-1]
            else:
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[d_voice[i]]
            seg['start'] = st
            seg['end'] = en
            seg['duration'] = en - st + 1
            voiceseg[i] = seg
    return voiceseg

def vad_specEN(data, wnd, inc, NIS, thr1, thr2, fs):
    from scipy.signal import medfilt
    #print('1',data.shape)
    x = enframe(data, wnd, inc)
    #print('2',x.shape)
    X = np.abs(np.fft.fft(x, axis=1))
    #print('3',X.shape)    ]
    if len(wnd) == 1:
        wlen = wnd
    else:
        wlen = len(wnd)
    df = fs / wlen
    fx1 = int(250 // df + 1)  # 250Hz位置    4
    fx2 = int(3500 // df + 1)  # 500Hz位置    44

    km = wlen // 8
    K = 0.5
    E = np.zeros((X.shape[0], wlen // 2))
    #print(E.shape)   #(1618,100)
    E[:, fx1 + 1:fx2 - 1] = X[:, fx1 + 1:fx2 - 1]
    #print('1',E.shape)   (1618, 100)
    E = np.multiply(E, E)
    #print('2',E.shape)    (1618, 100)
    Esum = np.sum(E, axis=1, keepdims=True)
    #print('3',Esum.shape)   (1618, 1)
    P1 = np.divide(E, Esum)
    E = np.where(P1 >= 0.9, 0, E)
    #print(E.shape)   (1618, 100)
    Eb0 = E[:, 0::4]
    Eb1 = E[:, 1::4]
    Eb2 = E[:, 2::4]
    Eb3 = E[:, 3::4]
    Eb = Eb0 + Eb1 + Eb2 + Eb3
    #print(Eb.shape)   #(1618, 25)
    prob = np.divide(Eb + K, np.sum(Eb + K, axis=1, keepdims=True))
    Hb = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
    #print(Hb.shape)   #(1618,)

    #求每帧的谱熵值
    for i in range(10):
        Hb = medfilt(Hb, 5)
    #print(Hb)   #(1618,)
    Me = np.mean(Hb)
    eth = np.mean(Hb[:NIS])
    Det = eth - Me
    T1 = thr1 * Det + Me
    T2 = thr2 * Det + Me
    # print(T1)
    # print(T2)
    voiceseg, vsl, SF, NF = vad_revr(Hb, T1, T2)
    return voiceseg, vsl, SF, NF, Hb

def vad_revr(dst1, T1, T2):
    """
    端点检测反向比较函数
    :param dst1:
    :param T1:
    :param T2:
    :return:
    """
    fn = len(dst1)
    maxsilence = 8
    minlen = 5
    status = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    xn = 0
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(1, fn):
        if status == 0 or status == 1:
            if dst1[n] < T2:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif dst1[n] < T1:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        if status == 2:
            if dst1[n] < T1:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF

def file_gothrough(path,suffix):
        # 遍历文件夹
        file_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file[-3:] == suffix:
                    file_list.append(os.path.join(root, file))
        numberOfSample = len(file_list)
        return file_list, numberOfSample

# path = '/Users/mss/Desktop/test_wave'
# file_list,_ = file_gothrough(path,'wav')
# print(file_list)
# num = 0
# for case in file_list:
data, fs = soundBase('test_data/13.wav').audioread()
#data, fs = soundBase('/Users/mss/Desktop/test1.wav').audioread()
# data1 = data

data = data - np.mean(data)
data /= np.max(data)
IS = 0.25
wlen = 200
inc = 80
N = len(data)
time = [i / fs for i in range(N)]
wnd = np.hamming(wlen)
overlap = wlen - inc
NIS = int((IS * fs - wlen) // inc + 1)
thr1 = 0.99
thr2 = 0.96
voiceseg, vsl, SF, NF, Enm = vad_specEN(data, wnd, inc, NIS, thr1, thr2, fs)
# print(voiceseg)
# start = (voiceseg[0]['start']-1)*inc+1
# end = (voiceseg[vsl-1]['end']-1)*inc+1
# soundBase(path='/Users/mss/Desktop/1.wav').audiowrite(data=data1[start:end], fs=fs, binary=False, channel=1)
#
y = np.zeros([1,len(data)])
for i in range(len(voiceseg)):
    start = (voiceseg[i]['start'] - 1) * inc + 1
    end = (voiceseg[i]['end'] - 1) * inc + 1
    y[0][start:end] = [1] * (end-start)

plt.subplot(2, 1, 1)
plt.plot(data)
plt.title('Time Signal')
plt.ylim([-1,1])

plt.subplot(2, 1, 2)
plt.plot(y[0])
plt.xlabel('frame')
plt.ylabel('Prob')

plt.tight_layout()
plt.show()

