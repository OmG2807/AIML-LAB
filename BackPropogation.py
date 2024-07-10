import numpy as np
x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
x=x/np.amax(x,axis=0)
y=y/100
def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivaties_sigmoid(x):
    return x*(1-x)
epoch, lr = 5000, 0.1
inputlayer_neurons, hiddenlayer_neurons, output_neurons = 2, 3, 1
wh, bh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons)), np.random.uniform(size=(1, hiddenlayer_neurons))
wout, bout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons)), np.random.uniform(size=(1, output_neurons))
for _ in range(epoch):
    hlayer_act = sigmoid(np.dot(x, wh) + bh)
    output = sigmoid(np.dot(hlayer_act, wout) + bout)  
    d_output = (y - output) * derivatives_sigmoid(output)
    d_hiddenlayer = d_output.dot(wout.T) * derivatives_sigmoid(hlayer_act)
    wout += hlayer_act.T.dot(d_output) * lr
    wh += x.T.dot(d_hiddenlayer) * lr
print(f"Input:\n{x}")
print(f"Actual Output:\n{y}")
print(f"Predicted Output:\n{output}")
