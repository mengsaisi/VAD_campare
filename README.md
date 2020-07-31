VAD
在工作过程中，如需要在有底噪的情况下检测残留回声、聚焦到有语音的部分进行数据分析等，都要用到语音端点检测，即用于鉴别音频信号当中的语音出现（speech presence）和语音消失（speech absence）。下面将对目前搜罗到的一些开源的VAD算法和一些书本上的VAD算法进行简单的算法原理学习、代码的实现及性能的分析。目的在于找出语音起始点和结束点切分准确的VAD算法。
此次主要对比一下几种VAD算法：
1.    双门限法
2.    相关法
3.    谱熵法
4.    能熵比法
5.    Python_vad(https://github.com/eesungkim/Voice_Activity_Detector/)。Github上开源的，试用表现好的VAD，基于论文《A Statistical Model-Based Voice Activity Detection》，论文下载地址https://wiki.inf.ed.ac.uk/twiki/pub/CSTR/ListenSemester1_2010_11/sohn_SPL99_statistical_model-based_VAD.pdf
6.    WebRTC_vad：从WebRTC中提取包装成python库，基于GMM的VAD.
7.    我的VAD

