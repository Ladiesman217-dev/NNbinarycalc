using System;

class NeuralNetwork
{
    private int inputSize, hiddenSize, outputSize;
    private double learningRate;
    private double[,] weights1, weights2;
    private double[] bias1, bias2;

    public NeuralNetwork(int input, int hidden, int output, double lr)
    {
        inputSize = input;
        hiddenSize = hidden;
        outputSize = output;
        learningRate = lr;

        weights1 = InitializeWeights(inputSize, hiddenSize);
        weights2 = InitializeWeights(hiddenSize, outputSize);
        bias1 = new double[hiddenSize];
        bias2 = new double[outputSize];
    }

    private double[,] InitializeWeights(int rows, int cols)
    {
        Random rand = new Random();
        double[,] weights = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                weights[i, j] = (rand.NextDouble() - 0.5) * 2;
        return weights;
        
        
    }

    private double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
    private double SigmoidDerivative(double x) => x * (1 - x);

    public double[] Predict(double[] inputs)
    {
        double[] hidden = Activate(MatrixMultiply(inputs, weights1, bias1));
        return Activate(MatrixMultiply(hidden, weights2, bias2));
    }

    private double[] MatrixMultiply(double[] inputs, double[,] weights, double[] biases)
    {
        int cols = weights.GetLength(1);
        double[] result = new double[cols];

        for (int j = 0; j < cols; j++)
        {
            result[j] = biases[j];
            for (int i = 0; i < inputs.Length; i++)
                result[j] += inputs[i] * weights[i, j];
        }
        return result;
    }

    private double[] Activate(double[] values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = Sigmoid(values[i]);
        return values;
    }

    public void Train(double[][] inputs, double[][] expectedOutputs, int iterations)
    {
        for (int iter = 0; iter < iterations; iter++)
        {
            for (int sample = 0; sample < inputs.Length; sample++)
            {
                // Forward propagation
                double[] hidden = Activate(MatrixMultiply(inputs[sample], weights1, bias1));
                double[] outputs = Activate(MatrixMultiply(hidden, weights2, bias2));

                // Calculate error
                double[] outputError = new double[outputSize];
                double[] outputDelta = new double[outputSize];
                for (int i = 0; i < outputSize; i++)
                {
                    outputError[i] = expectedOutputs[sample][i] - outputs[i];
                    outputDelta[i] = outputError[i] * SigmoidDerivative(outputs[i]);
                }

                // Hidden layer error and adjustment
                double[] hiddenError = new double[hiddenSize];
                double[] hiddenDelta = new double[hiddenSize];

                for (int i = 0; i < hiddenSize; i++)
                {
                    hiddenError[i] = 0;
                    for (int j = 0; j < outputSize; j++)
                        hiddenError[i] += outputDelta[j] * weights2[i, j];
                    
                    hiddenDelta[i] = hiddenError[i] * SigmoidDerivative(hidden[i]);
                }

                // Update weights and biases
                for (int i = 0; i < hiddenSize; i++)
                    for (int j = 0; j < outputSize; j++)
                        weights2[i, j] += learningRate * outputDelta[j] * hidden[i];

                for (int i = 0; i < outputSize; i++)
                    bias2[i] += learningRate * outputDelta[i];

                for (int i = 0; i < inputSize; i++)
                    for (int j = 0; j < hiddenSize; j++)
                        weights1[i, j] += learningRate * hiddenDelta[j] * inputs[sample][i];

                for (int i = 0; i < hiddenSize; i++)
                    bias1[i] += learningRate * hiddenDelta[i];
            }
        }
    }
}