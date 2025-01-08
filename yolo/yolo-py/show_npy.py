import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 加载 .npy 文件
data = np.load('./output/test_02_quant.npy')

print(data.shape)
print(max(data[0][6]))
# # 绘制热图
# sns.heatmap(data[0], cmap='viridis')
# plt.title('2D Array Heatmap')
# plt.show()