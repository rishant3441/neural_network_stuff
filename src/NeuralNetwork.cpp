#include "NeuralNetwork.hpp"

#include <iostream>

namespace NN
{
    BaseNetwork::BaseNetwork(int newLayout[], int layoutSize, const TrainingData& data, int epochs)
        : data(data), epochs(epochs)
    {
        layout.reserve(layoutSize);
        for (int i = 0; i < layoutSize; i++)
        {
            layout[i] = newLayout[i];
        }

        weights.resize(layout[1], layout[0]);
        bias.resize(layout[1], 1);

        weights.setRandom();
        bias.setRandom();
    }

    float BaseNetwork::sigmoidf(float x)
    {
        return 1 / (1 + std::exp(-x));
    }

    float BaseNetwork::cost(const wMatrix& weights, const bMatrix& bias, const TrainingData& data)
    {
        float result = 0.0f;
        for (int i = 0; i < layout[1]; i++)
        {
            for (int j = 0; j < data.inputs.size(); j++)
            {
                
                auto resultM = (weights * data.inputs[j]) + bias;
                auto sigM = resultM.unaryExpr(&BaseNetwork::sigmoidf);

                float d = sigM.coeff(i,0) - data.outputs[j].coeff(i, 0);
                result += d*d;
            }
        }
        latestCost = result / data.inputs.size();
        return result / data.inputs.size();
    }

    void BaseNetwork::derivative_cost(wMatrix weights, bMatrix bias, float eps, const TrainingData& data, wMatrix* dw, bMatrix* b)
    {
        float c = cost(weights, bias, data);
        float save = 0;

        wMatrix newW = weights;
        bMatrix newB = bias;

        for (int i = 0; i < weights.rows(); i++)
        {
            for (int j = 0; j < weights.cols(); j++)
            {
                save = weights.coeffRef(i, j);
                newW.coeffRef(i, j) += eps; // check this later
                dw->coeffRef(i, j) = (cost(newW, bias, data) - c)/eps;
                newW.coeffRef(i, j) = save; 
            }
        }

        for (int i = 0; i < bias.rows(); i++)
        {
            save = bias.coeffRef(i,0);
            newB.coeffRef(i, 0) += eps;
            b->coeffRef(i, 0) = (cost(weights, newB, data) - c)/eps;
            newB.coeffRef(i, 0) = save;
        }
    }

    void BaseNetwork::Learn(float eps, float rate)
    {
        for (int i = 0; i < epochs; i++)
        {
            float c = cost(weights, bias, data);
            wMatrix m{};
            bMatrix b{};

            m.resize(layout[1], layout[0]);
            b.resize(layout[1], 1);

            derivative_cost(weights, bias, eps, data, &m, &b);

            for (int i = 0; i < weights.rows(); i++)
            {
                for (int j = 0; j < weights.cols(); j++)
                {
                    weights.coeffRef(i, j) -= rate*(m.coeff(i, j));
                }
            }

            for (int i = 0; i < bias.rows(); i++)
            {
                bias(i,0) -= rate*(b.coeff(i,0));
            }

#ifdef NN_DEBUG
            PrintResults();
#endif
        }
    }

    void BaseNetwork::PrintResults()
    {
        std::cout << "Weights: " << weights << std::endl;
        std::cout << "Biases: " << bias << std::endl;

        for (int i = 0; i < data.outputs.size(); i++)
        {
            auto result = (weights * data.inputs[i]) + bias;
            auto sigM = result.unaryExpr(&sigmoidf);
            std::cout << "Result " << i << ": " << sigM << std::endl; 
        }

        std::cout << "Cost: " << (latestCost ? latestCost : cost(weights, bias, data)) << std::endl; 
    }
}