# 能熵比进行端点检测
import  numpy as np
from soundBase import *

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
    return frameout

def FrameTimeC(frameNum, frameLen, inc, fs):
    ll = np.array([i for i in range(frameNum)])
    return ((ll - 1) * inc + frameLen / 2) / fs

def STZcr(x, win, inc, delta=0):
    """
    计算短时过零率
    :param x:
    :param win:
    :param inc:
    :return:
    """
    absx = np.abs(x)
    x = np.where(absx < delta, 0, x)
    X = enframe(x, win, inc)
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    s = np.multiply(X1, X2)
    sgn = np.where(s < 0, 1, 0)
    return np.sum(sgn, axis=1)

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



def vad_forw(dst1, T1, T2):
    """
    端点检测正向比较函数
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
            if dst1[n] > T2:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif dst1[n] > T1:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        if status == 2:
            if dst1[n] > T1:
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

def vad_pro(data, wnd, inc, NIS, thr1, thr2, mode):
    """
    使用比例法检测端点
    :param data:
    :param wnd:
    :param inc:
    :param NIS:
    :param thr1:
    :param thr2:
    :param mode:
    :return:
    """
    from scipy.signal import medfilt
    x = enframe(data, wnd, inc)
    if len(wnd) == 1:
        wlen = wnd
    else:
        wlen = len(wnd)
    if mode == 1:  # 能零比
        a = 2
        b = 1
        LEn = np.log10(1 + np.sum(np.multiply(x, x) / a, axis=1))
        EZRn = LEn / (STZcr(data, wlen, inc) + b)
        for i in range(10):
            EZRn = medfilt(EZRn, 5)
        dth = np.mean(EZRn[:NIS])
        T1 = thr1 * dth
        T2 = thr2 * dth
        Epara = EZRn
    elif mode == 2:  # 能熵比
        a = 2
        X = np.abs(np.fft.fft(x, axis=1))
        X = X[:, :wlen // 2]
        Esum = np.log10(1 + np.sum(np.multiply(X, X) / a, axis=1))
        prob = X / np.sum(X, axis=1, keepdims=True)
        Hn = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
        Ef = np.sqrt(1 + np.abs(Esum / Hn))
        for i in range(10):
            Ef = medfilt(Ef, 5)
        Me = np.max(Ef)
        eth = np.mean(Ef[NIS])
        Det = Me - eth
        T1 = thr1 * Det + eth
        T2 = thr2 * Det + eth
        Epara = Ef
    voiceseg, vsl, SF, NF = vad_forw(Epara, T1, T2)
    return voiceseg, vsl, SF, NF, Epara

data, fs = soundBase('test_data/5.wav').audioread()
# data1 = data
# data = data - np.mean(data)
#data /= np.max(data)
IS = 0.25
wlen = 200
inc = 80
N = len(data)
time = [i / fs for i in range(N)]
wnd = np.hamming(wlen)
overlap = wlen - inc
NIS = int((IS * fs - wlen) // inc + 1)

mode = 2
if mode == 1:
    thr1 = 3
    thr2 = 4
    tlabel = '能零比'
elif mode == 2:
    thr1 = 0.05
    thr2 = 0.1
    tlabel = '能熵比'
voiceseg, vsl, SF, NF, Epara = vad_pro(data, wnd, inc, NIS, thr1, thr2, mode)

# start = (voiceseg[0]['start']-1)*inc+1
# end = (voiceseg[vsl-1]['end']-1)*inc+1
# soundBase(path='/Users/mss/Desktop/out_0.wav').audiowrite(data=data1[start:end], fs=fs, binary=False, channel=1)

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
