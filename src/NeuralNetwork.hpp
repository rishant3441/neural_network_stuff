#pragma once

#include <Eigen/Dense>

namespace NN
{
    struct TrainingData
    {
        using aMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

        TrainingData(int size, int layout[])
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
                output.resize(layout[1], 1);
            }
        }

        std::vector<aMatrix> inputs;
        std::vector<aMatrix> outputs;
    };

    class BaseNetwork
    {
    public:
        BaseNetwork(int newLayout[], int layoutSize, const TrainingData& data, int epochs = 10*1000);

        using aMatrix = Eigen::Matrix<float, Eigen::Dynamic, 1>;
        using wMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
        using bMatrix = Eigen::Matrix<float, Eigen::Dynamic, 1>;

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
}