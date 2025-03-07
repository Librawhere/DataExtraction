import matplotlib.pyplot as plt
import numpy as np


# loss
# training length is 80
# loss80 = [4.0114, 1.0523, 0.4153, 0.2381, 0.1833, 0.1560, 0.1393, 0.1282, 0.1203, 0.1140,
#           0.1094, 0.1056, 0.1028, 0.1001, 0.0980, 0.0960, 0.0947, 0.0933, 0.0921, 0.0912,
#           0.0903, 0.0893, 0.0885, 0.0879, 0.0872, 0.0867, 0.0861, 0.0856, 0.0851, 0.0846,
#           0.0841, 0.0837, 0.0833, 0.0828, 0.0825, 0.0821, 0.0817, 0.0814, 0.0811, 0.0808,
#           0.0804, 0.0800, 0.0798, 0.0795, 0.0793, 0.0789, 0.0786, 0.0784, 0.0781, 0.0778,
#           0.0775, 0.0773, 0.0771, 0.0768, 0.0766, 0.0764, 0.0762, 0.0761, 0.0759, 0.0757,
#           0.0756, 0.0754, 0.0753, 0.0752, 0.0751, 0.0751, 0.0750, 0.0750, 0.0749, 0.0749]
#
# loss120 = [3.8657, 0.7942, 0.2502, 0.1605, 0.1297, 0.1117, 0.0998, 0.0913, 0.0850, 0.0803,
#            0.0765, 0.0735, 0.0710, 0.0691, 0.0675, 0.0659, 0.0647, 0.0636, 0.0627, 0.0619,
#            0.0611, 0.0605, 0.0599, 0.0593, 0.0588, 0.0583, 0.0579, 0.0576, 0.0571, 0.0568,
#            0.0566, 0.0561, 0.0559, 0.0555, 0.0553, 0.0550, 0.0548, 0.0545, 0.0543, 0.0541,
#            0.0538, 0.0536, 0.0533, 0.0531, 0.0529, 0.0527, 0.0525, 0.0523, 0.0521, 0.0520,
#            0.0517, 0.0516, 0.0514, 0.0512, 0.0511, 0.0510, 0.0508, 0.0507, 0.0506, 0.0505,
#            0.0504, 0.0503, 0.0502, 0.0501, 0.0501, 0.0500, 0.0500, 0.0499, 0.0499, 0.0499]
#
# epochs = np.arange(70)
#
# plt.title('Loss of the training')
# plt.xlabel('Epochs')
# plt.ylabel('Loss value')
# plt.plot(epochs, loss80, color='r', label='Training sequence length is 80')
# plt.plot(epochs, loss120, color='b', label='Training sequence length is 120')
# plt.legend()
# pth = './results/{}.jpg'.format('loss')
# plt.savefig(pth)
# plt.close()


# Leakage

max_length = [50, 75, 100]
greedy_H_80 = [0.001, 0.001, 0.001]
greedy_S_80 = [0.471, 0.471, 0.471]
beam_H_80 = [0.001, 0.001, 0.001]
beam_S_80 = [0.512, 0.513, 0.514]

greedy_H_120 = [0.001, 0.001, 0.001]
greedy_S_120 = [0.476, 0.477, 0.477]
beam_H_120 = [0.001, 0.001, 0.001]
beam_S_120 = [0.525, 0.525, 0.524]

plt.title('Recall of the data leakage')
plt.ylabel('Recall')
plt.xlabel('Maximum of the generating length')

plt.plot(max_length, greedy_H_80, color='sandybrown', marker='o', label='GD_H_80')
plt.plot(max_length, greedy_S_80, color='sandybrown', marker='s', label='GD_S_80')
plt.plot(max_length, beam_H_80, color='r', marker='o', label='BS_H_80', linestyle='--')
plt.plot(max_length, beam_S_80, color='r', marker='s', label='BS_S_80', linestyle='--')

plt.plot(max_length, greedy_H_120, color='lightblue', marker='v', label='GD_H_120')
plt.plot(max_length, greedy_S_120, color='lightblue', marker='p', label='GD_S_120')
plt.plot(max_length, beam_H_120, color='b', marker='v', label='BS_H_120', linestyle='--')
plt.plot(max_length, beam_S_120, color='b', marker='p', label='BS_S_120', linestyle='--')

plt.legend()

pth = './results/{}.jpg'.format('recall')
plt.savefig(pth)
plt.close()




