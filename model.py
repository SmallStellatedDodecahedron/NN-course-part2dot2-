import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layer1=FullyConnectedLayer(n_input,hidden_layer_size);
        self.alayer1=ReLULayer();
        self.layer2=FullyConnectedLayer(hidden_layer_size,n_output);
        
        return None
        raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        p=self.params()
        p['1B'].grad=np.zeros_like(p['1B'].value)
        p['1W'].grad=np.zeros_like(p['1W'].value)
        p['2B'].grad=np.zeros_like(p['2B'].value)
        p['2W'].grad=np.zeros_like(p['2W'].value)
        
        #raise Exception("Not implemented!")
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        predictions=self.layer2.forward(self.alayer1.forward(self.layer1.forward(X)))
        loss, dsoftmaxcross =softmax_with_cross_entropy(predictions,y)
        dbw=self.layer1.backward(self.alayer1.backward(self.layer2.backward(dsoftmaxcross)))
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        R1B=l2_regularization(p['1B'].value,self.reg)
        R2B=l2_regularization(p['2B'].value,self.reg)
        R1W=l2_regularization(p['1W'].value,self.reg)
        R2W=l2_regularization(p['2W'].value,self.reg)
        p['1B'].grad+=R1B[1]
        p['2B'].grad+=R2B[1]
        p['1W'].grad+=R1W[1]
        p['2W'].grad+=R2W[1]
        """
        p['1B'].value-=(p['1B'].grad)
        p['2B'].value-=(p['2B'].grad)
        p['1W'].value-=(p['1W'].grad)
        p['2W'].value-=(p['2W'].grad)#"""
        regloss=R1B[0]+R2B[0]+R1W[0]+R2W[0]
        return loss+regloss
        raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], int)
        pred=self.layer2.forward(self.alayer1.forward(self.layer1.forward(X))).argmax().astype(int)
        return pred
        raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {}
        
        # TODO Implement aggregating all of the params
        result['1B']=self.layer1.params()['B'];
        result['2B']=self.layer2.params()['B'];
        result['1W']=self.layer1.params()['W'];
        result['2W']=self.layer2.params()['W'];
        return result
        raise Exception("Not implemented!")

        return result
