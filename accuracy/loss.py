import glob
import matplotlib.pyplot as plt

all_line = []
with open("../train_log/train_log3-96.txt", "r", encoding="utf-8") as f:
    lline = f.readlines()
    for ll in lline:
        all_line.append(ll)
dict = {}
index = 0
for line in all_line:
    if line.find("s/step - loss: ") > 0 and line.find(" - accuracy: ") > 0:
        value = line.split("s/step - loss: ")[-1].split(" - accuracy: ")[0].strip()
        print(value)
        dict[int(index)] = float(value)
        index += 1

test_data_1 = sorted(dict.items(), key=lambda x: x[0])
print(test_data_1)
x_axis = []
y_axis = []
for item in test_data_1:
    x_axis.append(item[0])
    y_axis.append(item[1])
plt.title('train loss')
plt.plot(x_axis[3:], y_axis[3:], color='green', label='training loss')
plt.grid()
plt.xlabel('step')
plt.ylabel('loss')
plt.show()
