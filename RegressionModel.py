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

class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()

        # Definir la arquitectura similar a la referencia
        self.layer1 = Linear(1, 32)
        self.layer2 = Linear(32, 64)
        self.layer3 = Linear(64, 32)
        self.layer4 = Linear(32, 1)
        
        # Inicializar pesos (similar a la inicialización Xavier en la referencia)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            torch.nn.init.xavier_normal_(layer.weight)
            

    
    def forward(self, x):
        """
        Runs the model for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Aplicar capas con activaciones ReLU entre ellas
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = relu(self.layer3(x))
        x = self.layer4(x)
        return x
   
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        # Obtener predicciones
        predictions = self.forward(x)
        
        # Calcular MSE como en la referencia
        return mse_loss(predictions, y)
       
    def train(self, dataset):
        """
        Trains the model.
        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader
        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
        """
        "*** YOUR CODE HERE ***"
        # Configurar el entrenamiento
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        batch_size = 32
        num_epochs = 2000
        target_loss = 0.02
        
        # Crear dataloader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Variables para seguimiento y early stopping
        best_loss = float('inf')
        patience = 0
        max_patience = 25
        
        # Loop de entrenamiento principal
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Iterar sobre lotes
            for batch in data_loader:
                x = batch['x']
                y = batch['label']
                
                # Poner gradientes a cero
                optimizer.zero_grad()
                
                # Calcular pérdida
                loss = self.get_loss(x, y)
                
                # Retropropagación
                loss.backward()
                
                # Actualizar pesos
                optimizer.step()
                
                # Acumular pérdida
                total_loss += loss.item()
                num_batches += 1
            
            # Calcular pérdida promedio
            avg_loss = total_loss / num_batches
            
            # Verificar pérdida objetivo
            if avg_loss <= target_loss:
                break
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                
            if patience >= max_patience:
                break