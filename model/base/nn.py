from tqdm import tqdm
import numpy as np

class Neuron:
    """Clase básica para entender una neurona"""

    def __init__(self, input_size):
        """
        Inicialización de una neuora con pesos y sesgos aleatorios

        Args:
            - input_size (int): Número de inputs para la neurona
        """
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.random.randn()

    def sigmoid(self, x):
        """Función de activación: Sigmoide.
        
        Args:
            - x (float): Valor numérico a aplicar.

        Returns:
            float: Valor de la sigmoide en `x`

        > **Nota**: En el caso de la neurona es la suma ponderada de
        > los inputs y pesos más el sesgo
        """
        return 1/(1 + np.exp(-x))

    def derivada_sigmoide(self, x):
        """Derivada respecto x de la sigmoide"""
        return x * (1 - x)
    
    def forward(self, inputs):
        """Función para calcular la salida de la neurona.
        
        Args:
            - inputs (np.ndarray(dtype=float)): Array de numpy con los inputs. 
        
        Returns:
            float: Output predicho
        """
        self.inputs = inputs.reshape(-1,1)

        ponderate_sum = np.dot(self.inputs.T, self.weights)
        self.total = ponderate_sum + self.bias

        self.output = self.sigmoid(self.total)
        return self.output.flatten()

    def backward(self, error, lr):
        """
        Función `backward` para actualizar los pesos y el sesgo de la neurona

        Args:
            - error (float): Error de la predicción.
            - lr (float): Ratio de aprendizaje de la neurona
        
        Returns:
            float: Error ajustado.
        """
        d_error = error * self.derivada_sigmoide(self.output)
        self.weights -= lr * d_error * self.inputs
        self.bias -= lr * d_error

        return d_error * self.weights
    
class Layer:
    """Clase que representa una capa. Una capa está formada por varias neuronas
    """

    def __init__(self, number_of_neurons, input_size):
        """
        Inicialización de una capa con múltiples neuronas:

        Args:
            - number_of_neurons (int): Número de neuronas.
            - input_size (int): Número de inputs que va a tener cada neurona.
        """
        self.neurons = [Neuron(input_size) for _ in range(number_of_neurons)]

    def forward(self,inputs):
        """
        Función para calcular la salida de la capa.

        Args:
            - inputs (np.ndarray(float)): Array 1D de numpy que entrará en la capa, se aplicará a todas las neuronas.

        Returns:
            - np.ndarray: Array con las salidas de todas las neuronas.
        """
        self.inputs = inputs.ravel()  
        self.outputs = np.array([neuron.forward(self.inputs) for neuron in self.neurons])
        return self.outputs

    def backward(self, errors, lr):
        """
        Función para propagar el error hacia atrás y actualizar pesos.
        """
        next_errors = np.zeros_like(self.inputs, dtype=np.float64)
        
        errors = errors.reshape(-1)
        for index, neuron in enumerate(self.neurons):
            grad = neuron.backward(errors[index], lr).flatten()
            next_errors += grad
        return next_errors

class NeuralNetwork:
    """Clase que representa una red de neuronas. Es decir
    un conjunto de capas"""
    
    def __init__(self, layer_sizes):
        """
        Inicializa la red neuronal con múltiples capas.

        Args:
            - layer_sizes (Iterable): Iterable con el tamaño de cada capa.
        """
        self.layers = [
            Layer(layer_sizes[index],layer_sizes[index - 1]) for index in range(1,len(layer_sizes))
        ]
    
    def forward(self, inputs):
        """
        Calcular el output. Propraga la entrada a través de todas las capas.

        Args:
            - inputs(np.ndarray): Array de Numpy con la entrada inicial.

        Returns:
            - np.ndarray: Salida final de la red neuronal. Se devuelve un output por neurona en la capa final
        """

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward(self, target, lr):
        """Propagación hacia atrás del error y actualización de pesos"""
        error = self.layers[-1].outputs - target
        for layer in reversed(self.layers):
            error = layer.backward(error,lr)

    def train(self, X, y, epochs, lr, patience):
        """
        Método para entrenar la red neuronal.
        """
        lowest_loss = float('inf')
        patience_counter = 0
        
        with tqdm(total=epochs, desc="Entrenando red", unit="epoch") as pbar:
            for epoch in range(epochs):
                total_loss = 0
                for x, t in zip(X, y):
                    pred = self.forward(x)
                    self.backward(t, lr)
                    total_loss += np.mean((t - pred)**2)
                
                avg_loss = total_loss / len(X)
                

                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'patience': f"{patience_counter}/{patience}"
                })
                pbar.update(1)

                if avg_loss < lowest_loss:
                    lowest_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter == patience:
                        pbar.set_postfix({
                            'loss': f"{avg_loss:.4f}", 
                            'status': 'Early Stopping!'
                        })
                        break
    
# Para probar la red neuronal
if __name__ == '__main__':

    nn = NeuralNetwork(layer_sizes=[2,4,1])

    # Datos XOR:
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    # Resultados sin entrenar:
    for index in range(len(X)):
        print(f"Entrada: {X[index]} - Salida predicha: {int(nn.forward(X[index]))} - Salida Real: {y[index]}")

    # Entrenamiento:
    nn.train(X, y, epochs=1000000, lr=0.01, patience=10)

    # Resultados después de entrenar:
    for index in range(len(X)):
        print(f"Entrada: {X[index]} - Salida predicha: {int(nn.forward(X[index]))} - Salida Real: {y[index]}")
