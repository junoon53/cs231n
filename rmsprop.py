import matplotlib.pyplot as plt
import numpy as np
import copy



v = np.array([0,0])
mu = 0.99
learning_rate = 0.5
dw_history = np.array([[-1,-5],[-1,5],[-1,-5],[-1,5],[-1,-5],[-1,5],[-1,-5],[-1,5],[-1,-5],[-1,5],[-1,-5],[-1,5],[-1,-5],[-1,5],[-1,-5]])

v_history = []
v_history.append(v)

def update_v(mu,learning_rate,dw):
    global v
    v = mu*v - learning_rate*dw
    v_history.append(copy.deepcopy(v))

for i in np.arange(len(dw_history)):
    update_v(mu,learning_rate,dw_history[i])

print v_history
print v


plt.plot(v_history)
plt.show()

