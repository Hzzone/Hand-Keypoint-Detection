import matplotlib.pyplot as plt
import re

with open('/Users/hzzone/Desktop/train.log') as f:
    data = f.read()

pattern = re.compile(r'''
I0418 (.*?)  5396 solver.cpp:243] Iteration (.*?), loss = (.*?)
I0418 (.*?)  5396 solver.cpp:259]     Train net output #0: mbox_loss = (.*?) \(\* 1 = (.*?) loss\)
I0418 (.*?)  5396 sgd_solver.cpp:138] Iteration (.*?), lr = (.*?)
''')
results = re.findall(pattern, data)
iter_num = []
total_loss = []
mbox_loss = []
learning_rate = []

for result in results:
    iter_num.append(int(result[1]))
    total_loss.append(float(result[2]))
    mbox_loss.append(float(result[4]))
    learning_rate.append(float(result[-1]))

plt.subplot(211)
plt.plot(iter_num, total_loss, iter_num, mbox_loss)
plt.legend(('total loss', 'mbox loss', 'learning rate'))
plt.subplot(212)
plt.plot(iter_num, learning_rate)
plt.legend('learning rate')

plt.show()

