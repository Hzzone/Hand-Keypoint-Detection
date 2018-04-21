import matplotlib.pyplot as plt
import re

with open('../model/train.log') as f:
    data = f.read()

pattern = re.compile(r'''
I0(.*?)solver.cpp:243] Iteration (.*?), loss = (.*?)
I0(.*?)solver.cpp:259]     Train net output #0: mbox_loss = (.*?) \(\* 1 = (.*?) loss\)
I0(.*?)sgd_solver.cpp:138] Iteration (.*?), lr = (.*?)
''')
results = re.findall(pattern, data)
iter_num = []
total_loss = []
mbox_loss = []
learning_rate = []
print(results)

for result in results:
    iter_num.append(int(result[1]))
    total_loss.append(float(result[2]))
    mbox_loss.append(float(result[4]))
    learning_rate.append(float(result[-1]))

plt.subplot(311)
plt.plot(iter_num, total_loss)
plt.subplot(312)
plt.plot(iter_num, mbox_loss)
plt.subplot(313)
plt.plot(iter_num, learning_rate)

plt.show()

