#include "NeuralNetwork.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <array>
#include <cassert>
#include <atomic>
#include <fstream>

#include <json.hpp>

#include <omp.h>

namespace NN
{
    using nlohmann::json;
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
        //return std::max(0.0f, x);
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
        
    MultiLayeredNetwork::MultiLayeredNetwork(int newLayout[], int layoutSize, ActF acts[], CostF costF)
        : data(1, newLayout, 0), costF(costF)
    {
        layout.reserve(layoutSize);
        for (int i = 0; i < layoutSize; i++)
        {
            layout.push_back(newLayout[i]);
        }

        activations.reserve(layoutSize - 1);
        for (int i = 0; i < layoutSize - 1; i++)
        {
            activations.push_back(acts[i]);
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

            switch (activations[i])
            {
                case ActF::SIGMOID:
                    sResult = newResult.unaryExpr(&sigmoidf);
                    break;
                case ActF::RELU:
                    sResult = newResult.unaryExpr(&reluf);
                    break;
                case ActF::SOFTMAX:
                    sResult = softmaxf(newResult);
                    break;
                default:
                    sResult = newResult.unaryExpr(&sigmoidf);
            }

            result.resize(sResult.rows(), sResult.cols());
            result = sResult;
        }        

        return result; 
    }

    float MultiLayeredNetwork::cost(const std::vector<wMatrix>& weights, const std::vector<bMatrix>& bias, const TrainingData& data)
    {
        std::atomic<float> finalResult = 0;
        #pragma omp parallel for
        for (int i = 0; i < data.inputs.size(); i++)
        {
            auto result = forward(weights, bias, data.inputs[i]);
            for (int j = 0; j < layout.back(); j++)
            {
                auto& E = data.outputs[i](j,0); // error
                auto& a = result(j,0); // activation
                float d;
                switch (costF)
                {
                    case CostF::MSE:
                        d = a - E; // activation - error 
                        finalResult += d*d;
                        break;
                    case CostF::CROSS_ENTROPY:
                        if (1-a == 0 || a == 0)
                            continue;
                        d = (E * std::log(a)) + (1 - E) * (log(1-a));
                        finalResult -= d;
                        break;
                }
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

    void MultiLayeredNetwork::Learn(const TrainingData& trainingData, int theEpochs, float eps, float rate)
    {
        data = trainingData;
        epochs = theEpochs;  
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
            std::cout << "Completed iteration: " << i+1 << " - Cost: " << latestCost << std::endl;
        }
    }

    void MultiLayeredNetwork::PrintResults()
    {
        std::cout << std::fixed;
        std::cout << std::setprecision(2); 
        for (int j = 0; j < layout.size() - 1; j++)
        {
            std::cout << "Weights: " << weightArray[j] << std::endl;
            std::cout << "Biases: " << biasArray[j] << std::endl;

            for (int i = 0; i < data.outputs.size(); i++)
            {
                auto result = forward(weightArray, biasArray, data.inputs[i]);
                std::cout << "Result " << i << ": " << (costF == CostF::CROSS_ENTROPY ? result * 100.0f : result) << std::endl; // not adaptive to all output types 
            }

            std::cout << "Cost: " << (latestCost ? latestCost : cost(weightArray, biasArray, data)) << std::endl; 
        }
        
    }

    float MultiLayeredNetwork::sigmoidf(float x)
    {
        return 1 / (1 + std::exp(-x));
    }

    float MultiLayeredNetwork::reluf(float x)
    {
        return std::max(0.01f * x, x);
    }

    NN::MultiLayeredNetwork::aMatrix MultiLayeredNetwork::softmaxf(const aMatrix& x)
    {
        float sum = 0;
        
        aMatrix out = x;
        for (unsigned int j = 0; j < x.rows(); j++)
        {
            sum += std::exp( x(j,0));
        }
        
        for (unsigned int i = 0; i < x.rows(); i++)
        {
            out(i,0) = std::exp( x(i,0) )/sum;
        }
        
        return out;
    }

    void MultiLayeredNetwork::Save(const std::string& filePath)
    {
        std::ofstream out(filePath);
        json array;

        /*std::vector<json> jsons;
        for (auto& weights : weightArray)
        {
            std::vector<double> weight(weights.data(), weights.data() + weights.rows() * weights.cols());
            array = weight;
            jsons.push_back(array);
        }*/

        for (int i = 0; i<weightArray.size(); i++)
        {
            std::vector<float> weight(weightArray[i].data(), weightArray[i].data() + weightArray[i].rows() * weightArray[i].cols());
            array["w" + std::to_string(i)] = weight;
        }

        for (int i = 0; i < biasArray.size(); i++)
        {

            std::vector<float> bias(biasArray[i].data(), biasArray[i].data() + biasArray[i].rows() * biasArray[i].cols());
            array["b" + std::to_string(i)] = bias;
        }

        out << std::setw(4) << array << std::endl;

        out.close();
    }

    void MultiLayeredNetwork::Load(const std::string& filePath)
    {
        std::ifstream in(filePath);
        json data = json::parse(in);

        int weightAmount = 0;
        int biasAmount = 0;
        for (int i = 0; i < biasArray.size(); i++)
        {
            std::vector<float> array = data["b" + std::to_string(i)];
            float* arrayPtr = array.data();

            biasArray[i] = Eigen::Map<wMatrix>(arrayPtr, biasArray[i].rows(), biasArray[i].cols());
        }
        for (int i = biasArray.size(); i < weightArray.size() + biasArray.size(); i++)
        {
            std::vector<float> array = data["w" + std::to_string(i - biasArray.size())];
            float* arrayPtr = array.data();

            weightArray[i - biasArray.size()] = Eigen::Map<wMatrix>(arrayPtr, weightArray[i - biasArray.size()].rows(), weightArray[i - biasArray.size()].cols());
        }
/*
        for (int i = 0; i < data.size(); i++)
        {
            std::vector<float> array = data["w" + std::to_string(i)];
            float* arrayPtr = array.data();

            weightArray[i] = Eigen::Map<wMatrix>(arrayPtr, weightArray[i].rows(), weightArray[i].cols());
        }
*/
        in.close();
    }
}