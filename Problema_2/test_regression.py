import torch
import numpy as np
import matplotlib.pyplot as plt
from models import RegressionModel

def test_regression_model():
    print("Generando datos de prueba...")
    
    # Crear datos para la función sin(x) en el rango [-2π, 2π]
    num_samples = 1000
    x_values = np.linspace(-2 * np.pi, 2 * np.pi, num_samples)
    y_values = np.sin(x_values)
    
    # Convertir a tensores de PyTorch
    x_tensor = torch.FloatTensor(x_values).reshape(-1, 1)
    y_tensor = torch.FloatTensor(y_values).reshape(-1, 1)
    # Guardar los datos en un archivo de CSV 
    np.savetxt('./Problema_2/datos_seno.csv', np.column_stack((x_values, y_values)), 
              delimiter=',', header='x,sin(x)', comments='')
    print("Datos guardados también en formato CSV como 'datos_seno.csv'")
    
    # Crear un dataset de PyTorch
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    
    # Crear instancia del modelo
    print("Inicializando el modelo...")
    model = RegressionModel()
    
    # Entrenar el modelo (con un learning_rate un poco más bajo para mayor estabilidad)
    print("Entrenando el modelo...")
    model.train_model(dataset, learning_rate=0.005, batch_size=64, num_epochs=3000, target_loss=0.02)
    
    # Evaluar el modelo
    print("Evaluando el modelo...")
    model.eval()  # Cambiar a modo de evaluación
    with torch.no_grad():
        predictions = model(x_tensor)
    
    # Calcular la pérdida final
    mse = ((predictions - y_tensor) ** 2).mean().item()
    print(f"Error cuadrático medio final: {mse:.6f}")
    
    # Verificar si cumple con el umbral requerido
    if mse <= 0.02:
        print("✅ El modelo cumple con el requisito de error ≤ 0.02")
    else:
        print("❌ El modelo no cumple con el requisito de error ≤ 0.02")
    
    # Visualizar los resultados
    print("Generando visualización...")
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'b-', label='sin(x) verdadero')
    plt.plot(x_values, predictions.numpy(), 'r--', label='Predicción del modelo')
    plt.legend()
    plt.title('Aproximación de sin(x) con Red Neuronal')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.grid(True)
    plt.savefig('./Problema_2/regresion_resultado.png')
    print("Gráfico guardado como 'regresion_resultado.png'")
    
    # Si matplotlib puede mostrar el gráfico, mostrarlo
    try:
        plt.show()
    except:
        print("No se pudo mostrar el gráfico (posible entorno sin interfaz gráfica)")

if __name__ == "__main__":
    test_regression_model()