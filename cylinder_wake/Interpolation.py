from scipy.interpolate import interp2d
import numpy as np
import matplotlib.pyplot as plt


from scipy.interpolate import griddata


# method 1
a = [1, 2, 3, 4]
b = [1, 2, 3]
ans = [4, 5, 6, 3, 4, 5, 2, 3, 4, 11, 12, 13]
A, B = np.meshgrid(a, b)
# print(A)
# print(B)
X_star = np.hstack((A.flatten()[:, None], B.flatten()[:, None]))
# print(X_star)

m = [1.5, 2.5]
n = [1.5, 2.5]
M, N = np.meshgrid(m, n)
U = griddata(X_star, ans, (M, N), method='cubic')
print(U)




# method 2
x=np.arange(1,11)
y=np.arange(1,11)

X,Y = np.meshgrid(x,y)
R = np.sqrt(X**2 +Y**2)
Z = np.sin(R)

print('X:\n',X)
print('Y:\n',Y)
print('Z:\n',Z)

fig1 = plt.figure(figsize=(9,8))
ax1 = fig1.add_subplot( projection='3d')
surf1 = ax1.plot_surface(X,Y,Z,cmap=plt.cm.viridis_r)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_xticks(range(1,11))
ax1.set_yticks(range(1,11))
ax1.set_title('Original graph',fontsize=15)
fig1.colorbar(surf1,shrink=0.5)
fig1.tight_layout()
plt.show()


# 使用kind='linear'方法对原始二维数据进行线性插值：
# 获取待插值的完整坐标点
x1 = np.arange(1,10.2,0.2)
y1 = np.arange(1,10.2,0.2)
X1,Y1 = np.meshgrid(x1,y1)

# 二维插值,kind='linear'
f1 = interp2d(x,y,Z,kind='linear')
Z1 = f1(x1,y1)
fig2 = plt.figure(figsize=(9,8))
ax2 = fig2.add_subplot( projection='3d')
surf2 = ax2.plot_surface(X1,Y1,Z1,cmap=plt.cm.viridis_r)
ax2.set_xlabel('X1')
ax2.set_ylabel('Y1')
ax2.set_zlabel('Z1')
ax2.set_xticks(range(1,11))
ax2.set_yticks(range(1,11))
ax2.set_title("kind='linear'",fontsize=15)
fig2.colorbar(surf2,shrink=0.5)
fig2.tight_layout()
plt.show()

# 使用kind='cubic'方法对原始二维数据进行三次式插值：
# 二维插值,kind='cubic'
f2 = interp2d(x,y,Z,kind='cubic')
Z2 = f2(x1,y1)
fig3 = plt.figure(figsize=(9,8))
ax3 = fig3.add_subplot( projection='3d')
surf3 = ax3.plot_surface(X1,Y1,Z2,cmap=plt.cm.viridis_r)
ax3.set_xlabel('X1')
ax3.set_ylabel('Y1')
ax3.set_zlabel('Z2')
ax3.set_xticks(range(1,11))
ax3.set_yticks(range(1,11))
ax3.set_title("kind='cubic'",fontsize=15)
fig3.colorbar(surf3,shrink=0.5)
fig3.tight_layout()
plt.show()

# 五次式插值(kind='quintic')
# 使用kind='quintic'方法对原始二维数据进行五次式插值：

f3 = interp2d(x,y,Z,kind='quintic')
Z3 = f3(x1,y1)
fig4 = plt.figure(figsize=(9,8))
ax4 = fig4.add_subplot( projection='3d')
surf4 = ax4.plot_surface(X1,Y1,Z3,cmap=plt.cm.viridis_r)
ax4.set_xlabel('X1')
ax4.set_ylabel('Y1')
ax4.set_zlabel('Z3')
ax4.set_xticks(range(1,11))
ax4.set_yticks(range(1,11))
ax4.set_title("kind='quintic'",fontsize=15)
fig4.colorbar(surf4,shrink=0.5)
fig4.tight_layout()
plt.show()