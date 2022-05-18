

##MY DEF
MNVAL=4.9406564584124654e-324
##--min val for log non zero
import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad
    raise Exception("Not implemented!")
    #return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    #def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    if len(predictions.shape)==1:   
        dprediction=predictions.copy()
        dprediction=(dprediction-np.max(dprediction))
        dprediction=np.exp(dprediction)
        dprediction=dprediction/np.sum(dprediction)
        loss=-np.log(dprediction[target_index])
        dprediction[target_index]=dprediction[target_index]-1
        return loss,dprediction
    
    
    
    dprediction=predictions.copy()
    dprediction=(dprediction.T-np.max(dprediction,axis=1)).T
    
    dprediction=np.exp(dprediction) 
    
    dsum=np.sum(dprediction,axis=1).reshape((len(dprediction),1))
    
    dprediction=dprediction/(dsum)+4.9406564584124654e-324 ##############
    
    targets=dprediction[np.arange(target_index.size),target_index.ravel()]
    #targets=np.choose(target_index.ravel(),dprediction.T)
    #print("q",dprediction,"a",dsum,"s");#raise Exception("AA")
    loss=-np.mean(np.log(targets))
    dprediction[np.arange(target_index.size),target_index.ravel()]=dprediction[np.arange(target_index.size),target_index.ravel()]-1;
    dprediction=dprediction/target_index.size
    
    return loss, dprediction
    raise Exception("Not implemented!")

    #return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.backS=0
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.backS=np.greater(X,0).astype(int)
        q= np.multiply(self.backS,X)
        #print("A\n",q)
        return q
        raise Exception("Not implemented!")

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result=np.multiply(self.backS,d_out)
        #print("B\n",d_result)
        return d_result
        raise Exception("Not implemented!")
        

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X=np.copy(X)
        return np.dot(self.X,self.W.value)+self.B.value
        raise Exception("Not implemented!")

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute
        
        
        self.W.grad=np.dot(self.X.T,d_out)
        self.B.grad=np.copy(d_out).sum(axis=0).reshape(self.B.value.shape);
        # It should be pretty similar to linear classifier from
        # the previous assignment
        
        return np.dot(d_out,self.W.value.T)
        raise Exception("Not implemented!")

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
