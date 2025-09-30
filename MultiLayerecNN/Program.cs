using System;
using System.Numerics;

class Program
{
    static void Main()
    {
        double number = Math.Pow(2, 2);
        
        Console.WriteLine(number);
        NeuralNetwork nn = new NeuralNetwork(3, 5, 2, 0.01);
        TrainNetwork(nn);

        bool continueQuerying = true;
        while (continueQuerying)
        {
            Console.WriteLine("\nEnter two binary numbers to add:");
            Console.Write("First number: ");
            string num1 = Console.ReadLine();
            Console.Write("Second number: ");
            string num2 = Console.ReadLine();

            int maxLength = Math.Max(num1.Length, num2.Length);
            num1 = num1.PadLeft(maxLength, '0');
            num2 = num2.PadLeft(maxLength, '0');

            string sumResult = "";
            string carryResult = "0"; // Carry starts at 0

            for (int i = maxLength - 1; i >= 0; i--)
            {
                double[] inputBits = { charToInt(num1[i]), charToInt(num2[i]), charToInt(carryResult[0]) };
                double[] outputBits = nn.Predict(inputBits);

                sumResult = $"{Math.Round(outputBits[0])}" + sumResult;

                // Update carry for the next iteration
                carryResult = $"{Math.Round(outputBits[1])}";
            }

            // Append carry if needed
            if (carryResult == "1") sumResult = carryResult + sumResult;

            Console.WriteLine($"Binary Sum: {sumResult}");

            Console.Write("\nContinue (y/n): ");
            string response = Console.ReadLine().ToLower();
            continueQuerying = (response == "y" || response == "yes");
        }


    }

    static int charToInt(char c) => c == '1' ? 1 : 0;

    static void TrainNetwork(NeuralNetwork nn)
    {
        double[][] inputs =
        {
            new double[] { 0, 0, 0 }, new double[] { 0, 0, 1 }, new double[] { 0, 1, 0 },
            new double[] { 0, 1, 1 }, new double[] { 1, 0, 0 }, new double[] { 1, 0, 1 },
            new double[] { 1, 1, 0 }, new double[] { 1, 1, 1 }
        };

        double[][] expectedOutputs =
        {
            new double[] { 0, 0 }, new double[] { 1, 0 }, new double[] { 1, 0 },
            new double[] { 0, 1 }, new double[] { 1, 0 }, new double[] { 0, 1 },
            new double[] { 0, 1 }, new double[] { 1, 1 }
        };

        nn.Train(inputs, expectedOutputs, 100000);
    }
}