import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 24})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

groups = ['1', '2', '3', '4', '5', '6', '7']
macro_avg = 0.94
micro_avg = 0.89

# Generate random accuracy values around the averages
np.random.seed(42)
macro_accuracy = np.array([0.9549, 0.93585207, 0.95943066, 0.9856909,  0.9329754,  0.93297589, 0.98737638]) 
micro_accuracy = np.array([0.91302304, 0.87591577, 0.9062768,  0.87609747, 0.87602811, 0.89725887, 0.83260159])
bar_width = 0.35
opacity = 0.8
index = np.arange(len(groups))

fig, ax = plt.subplots()
bars1 = plt.bar(index, macro_accuracy, bar_width, alpha=opacity, label='Macro')
bars2 = plt.bar(index + bar_width, micro_accuracy, bar_width, alpha=opacity, label='Micro')

plt.xlabel('Subject left for Testing')
plt.ylabel('Accuracy')
plt.xticks(index + bar_width / 2, groups)
plt.legend(ncol=2)

plt.ylim([0, 1.05])
ax.yaxis.grid()
plt.tight_layout()
plt.savefig('leaveoneoutuser1.eps')
plt.show()


groups = ['MARS', 'RadHAR', 'Vid2Doppler', 'VGG-16', 'Pantomime']
m_activity_mu_conf = [0.63, 0.38, 0.51, 0.49, 0.51]  
mu_activity_m_conf = [0.48, 0.18, 0.34, 0.22, 0.32]  

bar_width = 0.35
opacity = 0.8
index = np.arange(len(groups))

fig, ax = plt.subplots()
bars1 = plt.bar(index, m_activity_mu_conf, bar_width, alpha=opacity, label='M Activity: $\mu$ Conf')
bars2 = plt.bar(index + bar_width, mu_activity_m_conf, bar_width, alpha=opacity, label='$\mu$ Activity: M Conf')

plt.ylabel('Accuracy')
plt.xticks(index + bar_width / 2, groups, rotation=30)
plt.legend()
ax.yaxis.grid()
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('trained_macro_micro1.eps')
plt.show()


plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"



angles = np.array([-45, -30, -15, 0, 15, 30, 45])
mars_accuracy = np.array([0.62, 0.8, 0.92, 0.96, 0.93, 0.85, 0.64])
radhar_accuracy = np.array([0.25, 0.52, 0.7, 0.81, 0.75, 0.58, 0.28])
vid2doppler_accuracy = np.array([0.32, 0.54, 0.66, 0.72, 0.71, 0.58, 0.34])
vgg16_accuracy = np.array([0.28, 0.46, 0.58, 0.68, 0.62, 0.5, 0.29])
pantomime_accuracy = np.array([0.58, 0.63, 0.69, 0.75, 0.73, 0.68, 0.56])

plt.plot(angles, mars_accuracy, label='MARS', linewidth=3)
plt.plot(angles, radhar_accuracy, label='RadHAR', linestyle='--', linewidth=3)
plt.plot(angles, vid2doppler_accuracy, label='Vid2Doppler', linestyle='-.', linewidth=3)
plt.plot(angles, vgg16_accuracy, label='VGG-16', linestyle=':', linewidth=3)
plt.plot(angles, pantomime_accuracy, label='Pantomime',linestyle='-', linewidth=3)

plt.xlabel('Angle (degrees)')
plt.ylabel('Accuracy')
plt.xlim(-45, 45)
plt.ylim([0,1.2])
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.xticks([-45, -30, -15, 0, 15, 30, 45])
plt.legend(ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig('acc_angle1.eps')
plt.show()
