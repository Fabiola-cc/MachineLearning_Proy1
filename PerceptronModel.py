from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim

class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'
        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        weight_vector = torch.ones(1, dimensions, dtype=torch.float32) 
        self.w = Parameter(weight_vector, requires_grad=False) 

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        return torch.matmul(self.w, x.T)
        

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if self.run(x) >= 0 else -1



    def train(self, dataset):
        with torch.no_grad():  # No usamos autograd, actualizamos los pesos manualmente
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  
        
            converged = False  
            while not converged:  
                converged = True  # Suponemos que convergerá
            
                for data in dataloader:  
                    x, y = data['x'], data['label'].item()  # Obtenemos x y etiqueta
                
                    prediction = self.get_prediction(x)
                    if prediction != y:  # Si está mal clasificado
                        self.w += y * x.squeeze(0)  # Actualizamos pesos
                        converged = False  # Si hubo una actualización, no ha convergido


