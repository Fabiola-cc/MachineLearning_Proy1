import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RegressionModel(nn.Module):
    """
    Una red neuronal para aproximar la función sin(x) en el intervalo [-2π, 2π]
    """
    
    def __init__(self):
        """
        Inicializa las capas y parámetros del modelo.
        """
        super(RegressionModel, self).__init__()
        
        # Definir la arquitectura para aproximar sin(x)
        self.model = nn.Sequential(
            nn.Linear(1, 32),  # Primera capa: 1 entrada -> 32 neuronas
            nn.ReLU(),         # Activación ReLU para introducir no-linealidad
            nn.Linear(32, 64), # Segunda capa: 32 -> 64 neuronas
            nn.ReLU(),         # Activación ReLU
            nn.Linear(64, 32), # Tercera capa: 64 -> 32 neuronas
            nn.ReLU(),         # Activación ReLU
            nn.Linear(32, 1)   # Capa de salida: 32 -> 1 (predicción)
        )
        
        # Inicializar los pesos con la inicialización Xavier
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        """
        Método forward para PyTorch. Pasa los datos a través del modelo.
        
        Args:
            x: Tensor de entrada de forma [batch_size, 1]
        
        Returns:
            Tensor de salida de forma [batch_size, 1] con las predicciones
        """
        return self.model(x)

    # Alias para forward para mantener compatibilidad con el código proporcionado
    def run(self, x):
        """
        Pasa los datos a través del modelo y devuelve las predicciones.
        
        Args:
            x: Tensor de entrada de forma [batch_size, 1]
        
        Returns:
            Tensor de salida de forma [batch_size, 1] con las predicciones
        """
        return self.forward(x)
    

    def get_loss(self, x, y):
        """
        Calcula la pérdida MSE entre las predicciones del modelo y los valores objetivo.
        
        Args:
            x: Tensor de entrada de forma [batch_size, 1]
            y: Tensor objetivo de forma [batch_size, 1]
        
        Returns:
            Pérdida MSE (error cuadrático medio)
        """
        # Obtener las predicciones pasando x a través de la red
        predictions = self.run(x)
        
        # Calcular el MSE (Mean Squared Error)
        loss_fn = nn.MSELoss()
        return loss_fn(predictions, y)

    def train(self, dataset, learning_rate=0.01, batch_size=32, num_epochs=2000, target_loss=0.02):
        """
        Entrena el modelo usando actualizaciones basadas en gradientes.
        
        Args:
            dataset: Dataset que contiene los datos de entrenamiento
            learning_rate: Tasa de aprendizaje para el optimizador
            batch_size: Tamaño del lote para el entrenamiento
            num_epochs: Número máximo de épocas para entrenar
            target_loss: Pérdida objetivo para detener el entrenamiento
        """
        # Crear el optimizador Adam
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Crear el dataloader para manejar los batches
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Inicializar variables para seguimiento y early stopping
        best_loss = float('inf')
        patience = 0
        max_patience = 25  # Épocas para esperar mejora antes de detener
        
        # Loop de entrenamiento principal
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Iterar sobre los lotes del dataset
            for x, y in data_loader:
                # Poner a cero los gradientes antes de cada actualización
                optimizer.zero_grad()
                
                # Calcular la pérdida para el batch actual
                loss = self.get_loss(x, y)
                
                # Retropropagación: calcular gradientes
                loss.backward()
                
                # Actualizar pesos usando el optimizador
                optimizer.step()
                
                # Acumular pérdida para calcular el promedio
                total_loss += loss.item()
                num_batches += 1
            
            # Calcular pérdida promedio de la época
            avg_loss = total_loss / num_batches
            
            # Imprimir progreso periódicamente
            if epoch % 100 == 0:
                print(f"Época {epoch}/{num_epochs}, Pérdida: {avg_loss:.6f}")
            
            # Verificar si hemos alcanzado la pérdida objetivo
            if avg_loss <= target_loss:
                print(f"¡Pérdida objetivo alcanzada! Pérdida: {avg_loss:.6f}")
                break
            
            # Lógica de early stopping para evitar sobreentrenamiento
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                
            if patience >= max_patience:
                print(f"Early stopping después de {max_patience} épocas sin mejora")
                break
                
        print(f"Entrenamiento completado. Mejor pérdida: {best_loss:.6f}")
