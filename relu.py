import numpy as np

def fprop(inputs):
	# Element wise comparsion with 0
	zeros = np.zeros((inputs.shape))
	return np.maximum(zeros,inputs)
	# Does forward propagation

def bprop(grads_outputs, inputs):
	# Returns gradient with respect to inputs
	inputs[inputs>0] = 1
	inputs[inputs<0] = 0
	return grads_outputs * inputs



test_inputs = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]])
test_relu_outputs = np.array([[0.1, 0., 0.3], [0., 0.5, 0.]])
test_grads_wrt_outputs = np.array([[5., 10., -10.], [-5., 0., 10.]])
test_relu_grads_wrt_inputs = np.array([[5., 0., -10.], [-0., 0., 0.]])


print fprop(test_inputs)
print test_relu_outputs

print bprop(test_grads_wrt_outputs, test_inputs)
print test_relu_grads_wrt_inputs
