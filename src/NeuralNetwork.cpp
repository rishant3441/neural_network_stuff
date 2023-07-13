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

            PrintResults();
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

            //PrintResults(); // -- uncomment to print each epoch's results
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
        
    MultiLayeredNetwork::MultiLayeredNetwork(int newLayout[], int layoutSize, const TrainingData& data, int epochs)
        : data(data), epochs(epochs)
    {
        layout.reserve(layoutSize);
        for (int i = 0; i < layoutSize; i++)
        {
            layout.push_back(newLayout[i]);
        }

        for (int i = 0; i < layoutSize - 1; i++)
        {
            weightArray.push_back({});
            biasArray.push_back({});

            weightArray[i].resize(layout[i+1],layout[i]);
            biasArray[i].resize(layout[i+1],1);

            weightArray[i].setRandom();
            biasArray[i].setRandom();
        }
    }

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MultiLayeredNetwork::forward(const std::vector<wMatrix>& weights, const std::vector<bMatrix>& bias, const aMatrix& input)
    {
        dMatrix result{};
        result.resize(input.rows(), input.cols());
        result = input;

        for (int i = 0; i < layout.size() - 1; i++)
        {
            auto newResult = (weights[i] * result) + bias[i];
            dMatrix sResult;
            sResult.resize(weights[i].rows(), result.cols());
            sResult = newResult.unaryExpr(&MultiLayeredNetwork::sigmoidf);


            result.resize(sResult.rows(), sResult.cols());
            result = sResult;
        }        

        return result; 
    }

    float MultiLayeredNetwork::cost(const std::vector<wMatrix>& weights, const std::vector<bMatrix>& bias, const TrainingData& data)
    {
        float finalResult = 0;
        for (int i = 0; i < data.inputs.size(); i++)
        {
            auto result = forward(weights, bias, data.inputs[i]);
            for (int j = 0; j < layout.back(); j++)
            {
                float d = result(j,0) - data.outputs[i](j,0); 
                finalResult += d*d;
            }
        }
        latestCost = finalResult/data.inputs.size();
        return finalResult/data.inputs.size();
    }

    void MultiLayeredNetwork::derivative_cost(float eps, std::vector<wMatrix>* weightP, std::vector<bMatrix>* biasP)
    {
        float c = cost(weightArray, biasArray, data);
        float saved = 0;

        std::vector<wMatrix> newW;
        std::vector<bMatrix> newB;

        for (int i = 0; i < layout.size() - 1; i++)
        {
            newW.push_back({});
            newB.push_back({});

            newW[i].resize(layout[i+1],layout[i]);
            newB[i].resize(layout[i+1],1);

            newW[i] = weightArray[i];
            newB[i] = biasArray[i];
        }

        for (int i = 0; i < layout.size() - 1; i++)
        {
            for (int j = 0; j < weightArray[i].rows(); j++)
            {
                for (int k = 0; k < weightArray[i].cols(); k++)
                {
                    saved = weightArray[i](j,k);
                    newW[i](j,k) += eps;
                    (*weightP)[i](j,k) = (cost(newW, biasArray, data)-c)/eps;
                    newW[i](j,k) = saved;
                }
            }

            for (int j = 0; j < biasArray[i].rows(); j++)
            {
                saved = biasArray[i](j,0);
                newB[i](j,0) += eps;
                (*biasP)[i](j,0) = (cost(weightArray, newB, data)-c)/eps;
                newB[i](j,0) = saved;
            }
        }
    }

    void MultiLayeredNetwork::Learn(float eps, float rate)
    {
        for (int i = 0; i < epochs; i++)
        {
            float c = cost(weightArray, biasArray, data);

            std::vector<wMatrix> newW;
            std::vector<bMatrix> newB;

            for (int i = 0; i < layout.size() - 1; i++)
            {
                newW.push_back({});
                newB.push_back({});

                newW[i].resize(layout[i+1],layout[i]);
                newB[i].resize(layout[i+1],1);

                newW[i] = weightArray[i];
                newB[i] = biasArray[i];
            }

            derivative_cost(eps, &newW, &newB);

            for (int j = 0; j < newW.size(); j++)
            {
                for (int k = 0; k < newW[j].rows(); k++)
                {
                    for (int l = 0; l < newW[j].cols(); l++)
                    {
                        weightArray[j](k,l) -= rate*(newW[j](k,l));
                    }
                }
                for (int k = 0; k < newB[j].rows(); k++)
                {
                    biasArray[j](k,0) -= rate*(newB[j](k,0));
                }
            }
        }
    }

    void MultiLayeredNetwork::PrintResults()
    {
        for (int j = 0; j < layout.size() - 1; j++)
        {
            std::cout << "Weights: " << weightArray[j] << std::endl;
            std::cout << "Biases: " << biasArray[j] << std::endl;

            for (int i = 0; i < data.outputs.size(); i++)
            {
                auto result = forward(weightArray, biasArray, data.inputs[i]);
                std::cout << "Result " << i << ": " << result(0,0) << std::endl; // not adaptive to all output types 
            }

            std::cout << "Cost: " << (latestCost ? latestCost : cost(weightArray, biasArray, data)) << std::endl; 
        }
        
    }

    float MultiLayeredNetwork::sigmoidf(float x)
    {
        return 1 / (1 + std::exp(-x));
    }
}