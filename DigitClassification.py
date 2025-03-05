from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module

import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim


class DigitClassificationModel(Module):
    """
    Modelo para la clasificación de dígitos escritos a mano utilizando el conjunto de datos MNIST.
    
    Cada imagen de dígito es una imagen en escala de grises de 28x28 píxeles, que se aplana
    en un vector de 784 dimensiones para este modelo. Cada entrada en el vector es un número flotante entre 0 y 1.

    El objetivo es clasificar cada dígito en una de las 10 clases (números del 0 al 9).
    """
    def __init__(self):
        super().__init__()
        input_size = 28 * 28  # 784 píxeles
        hidden_size = 200  # Tamaño de la capa oculta
        output_size = 10  # 10 clases de dígitos

        # Primera capa lineal: entrada a capa oculta
        self.linear1 = Linear(input_size, hidden_size)
        
        # Segunda capa lineal: capa oculta a salida
        self.linear2 = Linear(hidden_size, output_size)

    def run(self, x):
        """
        Ejecuta el modelo para un lote de ejemplos.

        El modelo debe predecir un tensor con forma (batch_size x 10),
        que contenga puntuaciones. Puntuaciones más altas corresponden
        a una mayor probabilidad de que la imagen pertenezca a una clase en particular.

        Entradas:
            x: un tensor con forma (batch_size x 784)
        Salida:
            Un tensor con forma (batch_size x 10) que contiene los logits predichos.
        """
        # Primera capa con activación ReLU
        hidden = relu(self.linear1(x))
        
        # Segunda capa sin activación (devuelve logits)
        return self.linear2(hidden)

    def get_loss(self, x, y):
        """
        Calcula la pérdida para un lote de ejemplos.
        
        Las etiquetas correctas `y` están representadas como un tensor con forma
        (batch_size x 10). Cada fila es un vector one-hot codificando la clase correcta
        del dígito (0-9).

        Entradas:
            x: un tensor con forma (batch_size x 784)
            y: un tensor con forma (batch_size x 10)
        Retorno:
            Un tensor con la pérdida calculada.
        """
        # Calcula la pérdida de entropía cruzada entre las predicciones y las etiquetas reales
        predictions = self.run(x)
        return cross_entropy(predictions, y)
        
    def train(self, dataset):
        """
        Entrena el modelo utilizando un conjunto de datos.
        """
        # Hiperparámetros
        batch_size = 100
        learning_rate = 0.5
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        # Número total de ejemplos de entrenamiento
        total_examples = len(dataset.x)

        # Bucle de entrenamiento
        for epoch in range(10):  # Número típico de épocas
            # Itera sobre el conjunto de datos en lotes
            for start in range(0, total_examples, batch_size):
                # Listas para almacenar el lote actual
                x_batch = []
                y_batch = []

                # Recopilar ejemplos del lote actual
                for i in range(start, min(start + batch_size, total_examples)):
                    item = dataset[i]
                    x_batch.append(item['x'])
                    y_batch.append(item['label'])

                # Convertir las listas en tensores
                x_batch = stack(x_batch)  # Apila los tensores en una nueva dimensión
                y_batch = stack(y_batch)

                # Reiniciar gradientes acumulados
                optimizer.zero_grad()

                # Calcular la pérdida
                loss = self.get_loss(x_batch, y_batch)

                # Retropropagación
                loss.backward()

                # Actualizar los pesos del modelo
                optimizer.step()

            # Comprobar la precisión de validación
            validation_accuracy = dataset.get_validation_accuracy()
            
            # Detener el entrenamiento si la precisión es lo suficientemente alta
            if validation_accuracy > 0.97:
                break