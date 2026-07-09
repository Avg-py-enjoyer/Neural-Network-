#importing libraries
import numpy as np
import matplotlib.pyplot as plt

#initializing parameters
def init_parameters(layer_dim):
    np.random.seed(3)
    params = {}
    L =len(layer_dim)

    for l in range(1,L):
        params['W'+str(l)]=np.random.randn(layer_dim[l],layer_dim[l-1])*0.01
        params['b'+str(l)]=np.zeros((layer_dim[l],1))

    return params

#sigmoid activation
def sigmoid(z):
    A = 1/(1+np.exp(np.dot(-1,z)))
    cache = (z)

    return A, cache

#forward propogation code
def forward_propogation(X,params):
    #X is the input
    A = X
    caches=[]
    L=len(params)//2

    for l in range(1, L+1):
        A_prev = A

        #linear hypothesis
        Z = np.dot(params['W'+str(l),A_prev])+params['b'+str(l)]

        linear_cache = (A_prev,params['W'+str(l)],params['b'+str(l)])

        #applying activation function
        A, activation_cache = sigmoid(Z)


        cache = (linear_cache,activation_cache)
        caches.append(cache)

    return A, caches

#defining cost function
def cost_function(A,Y):
    m = Y.shape[1]

    cost = (-1/m)*(np.dot(np.log(A),Y.T)) + np.dot(np.log(1-A),(1-Y).T)

    return cost

#single layer backpropogation
def single_back_layer(dA, cache):
    linear_cache, activation_cache = cache

    Z= activation_cache

    dZ = dA*sigmoid(Z)*(1-sigmoid(Z)) #derivative of sigmoid function

    A_prev,W,b = linear_cache
    m= A_prev.shape[1]

    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)

    return dA_prev,dW,db

#backpropagtion
def backprop(AL,Y,caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -np.divide(Y,AL)-np.divide(1-Y, 1-AL)

    current_cache = caches[L-1]
    grads['dA'+str(L-1)],grads['dW'+str(L-1)],grads['db'+str(L-1)] = single_back_layer(dAL,current_cache)

    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp,dW_temp,db_temp = single_back_layer(grads['dA'+str(l),current_cache])
        grads['dA'+str(l)]=dA_prev_temp
        grads['dW'+str(l+1)]=dW_temp
        grads['db'+str(l+1)]=db_temp 

    return grads

#update parameters
def update_parameters(grads,parameters,learning_rate):
    L=len(parameters)//2

    for l in range(L):
        parameters['W'+str(l+1)]=parameters['W'+str(l+1)] - learning_rate*grads['W'+str(l+1)]     
        parameters['b'+str(l+1)]=parameters['b'+str(l+1)] - learning_rate*grads['b'+str(l+1)]

    return parameters

#training NN

def train(X,Y,layer_dims,epochs,lr):
    params= init_parameters(layer_dims)
    cost_history=[]

    for i in range(epochs):
        Y_hat,caches = forward_propogation(X,params)
        cost = cost_function(Y_hat,Y)
        cost_history.append(cost)
        grads = backprop(Y_hat,Y,caches)

        params = update_parameters(params,grads,lr)

    return params,cost_history
