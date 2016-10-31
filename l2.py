import numpy as np

def calc(b,parameter):
	return b * 0.5 *  np.sum(parameter * parameter)


def grad(b,parameter):
	return b * parameter


test_params_1 = np.array([[0.5, 0.3, -1.2, 5.8], [0.2, -3.1, 4.9, -5.0]])
true_l1_grad_1 = np.array([[0.25, 0.15, -0.6, 2.9], [0.1, -1.55, 2.45, -2.5]]) 
b = 0.5
print calc(b,test_params_1)
print np.allclose(true_l1_grad_1, grad(b,test_params_1))

