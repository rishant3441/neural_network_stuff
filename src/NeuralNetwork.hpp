#pragma once

#include <Eigen/Dense>

namespace NN
{
    struct TrainingData
    {
        using aMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

        TrainingData(int size, int layout[], int lastIndex)
        {
            for (int i = 0; i < size; i++)
            {
                inputs.push_back({});
                outputs.push_back({});
            }

            for (auto& input : inputs)
            {
                input.resize(layout[0], 1);
            }

            for (auto& output : outputs)
            {
                output.resize(layout[lastIndex], 1);
            }
        }

        std::vector<aMatrix> inputs;
        std::vector<aMatrix> outputs;
    };

    enum class ActF
    {
        SIGMOID = 0, RELU, SOFTMAX
    };

    enum class CostF
    {
        MSE = 0, CROSS_ENTROPY
    };

    // Layout in the form of [first layer # of neurons, 2nd layer, etc.]
    // Currently only works with a 2 layer system (input and output)
    class BaseNetwork
    {
    public:
        BaseNetwork(int newLayout[], int layoutSize, const TrainingData& data, int epochs = 10*1000);

        using aMatrix = Eigen::Matrix<float, Eigen::Dynamic, 1>;                // Activation Matrix/Vector
        using wMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;   // Weight Matrix
        using bMatrix = Eigen::Matrix<float, Eigen::Dynamic, 1>;                // Bias Matrix/Vector

        void Learn(float eps = 1e-1, float rate = 1e-1);
        void PrintResults();

    private:
        float cost(const wMatrix& weights, const bMatrix& bias, const TrainingData& data);
        void derivative_cost(wMatrix weights, bMatrix bias, float eps, const TrainingData& data, wMatrix* dw, bMatrix* b);

        static float sigmoidf(float x);

        int epochs;
        TrainingData data;
        std::vector<int> layout;

        wMatrix weights; 
        bMatrix bias;

        float latestCost = 0;
    };

    class MultiLayeredNetwork
    {
    public:
        MultiLayeredNetwork(int newLayout[], int layoutSize, ActF acts[], CostF costF = CostF::MSE);

        using aMatrix = Eigen::Matrix<float, Eigen::Dynamic, 1>;                // Activation Matrix/Vector
        using wMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;   // Weight Matrix
        using bMatrix = Eigen::Matrix<float, Eigen::Dynamic, 1>;                // Bias Matrix/Vector
        using dMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;   // Dynamic Matrix

        void Learn(const TrainingData& trainingData, int theEpochs = 100*100, float eps = 1e-1, float rate = 1e-1);
        void PrintResults();

        void Save(const std::string& filePath);
        void Load(const std::string& filePath);

    private:
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> forward(const std::vector<wMatrix>& weights, const std::vector<bMatrix>& bias, const aMatrix& input);

        float cost(const std::vector<wMatrix>& weights, const std::vector<bMatrix>& bias, const TrainingData& data);
        void derivative_cost(float eps, std::vector<wMatrix>* weightP, std::vector<bMatrix>* biasP);

        static float sigmoidf(float x);
        static float reluf(float x);
        static aMatrix softmaxf(const aMatrix& x);

        std::vector<wMatrix> weightArray;
        std::vector<bMatrix> biasArray;        

        std::vector<int> layout;
        std::vector<ActF> activations;

        TrainingData data;
        int epochs;
        float latestCost = 0;
        CostF costF;
    };
}