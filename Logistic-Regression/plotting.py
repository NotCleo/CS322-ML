import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt('/home/amrut/Downloads/logisticX.csv', delimiter=',')
y = np.loadtxt('/home/amrut/Downloads/logisticY.csv', delimiter=',')

X_class0 = X[y == 0]
X_class1 = X[y == 1]

plt.figure(figsize=(8, 6))
plt.scatter(X_class0[:, 0], X_class0[:, 1], c='red', marker='o', label='y = 0', edgecolors='black')
plt.scatter(X_class1[:, 0], X_class1[:, 1], c='blue', marker='^', label='y = 1', edgecolors='black')

# the weights from terminal output
# Gradient Descent Weights
theta_gd = np.array([0.028465, 1.939959, -1.902999])
# Newton's Method Weights
theta_nt = np.array([0.223282, 1.962615, -1.964858])

x_vals = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])

y_vals_gd = -(theta_gd[1] / theta_gd[2]) * x_vals - (theta_gd[0] / theta_gd[2])
y_vals_nt = -(theta_nt[1] / theta_nt[2]) * x_vals - (theta_nt[0] / theta_nt[2])

plt.plot(x_vals, y_vals_gd, color='blue', linestyle='--', linewidth=2, label='Gradient Descent')
plt.plot(x_vals, y_vals_nt, color='black', linestyle='-', linewidth=2, label="Newton's Method")

plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('/home/amrut/decision_boundaries.png', dpi=300)
plt.show()
