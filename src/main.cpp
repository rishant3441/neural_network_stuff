#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <array>

#define NN_DEBUG
#include "NeuralNetwork.hpp"

using aMatrix = Eigen::Matrix<float, 2, 1>;
using wMatrix = Eigen::Matrix<float, 1, 2>;
using bMatrix = Eigen::Matrix<float, 1, 1>;

struct TrainingData
{
    std::array<aMatrix, 4> activations;
    std::array<float, 4> results;
};

float sigmoidf(float x)
{
    return 1 / (1 + std::exp(-x));
}

float cost(wMatrix weights, bMatrix bias, const TrainingData& data)
{
    float result = 0.0f;
    for (int i = 0; i < 4; i++)
    {
        auto resultM = (weights * data.activations[i]) + bias;
        auto sigM = resultM.unaryExpr(&sigmoidf);

        float d = sigM.coeff(0,0) - data.results[i];
        result += d*d;
    }
    return result / 4;
}

void derivative_cost(wMatrix weights, bMatrix bias, float eps, const TrainingData& data, wMatrix* dw, bMatrix* b)
{
    float c = cost(weights, bias, data);
    float w1 = weights.coeffRef(0,0);
    float w2 = weights.coeffRef(0,1); 
    float bF = bias.coeffRef(0,0);

    wMatrix m1;
    m1 << w1 + eps, w2;
    wMatrix m2;
    m2 << w1, w2 + eps;
    bMatrix b1;
    b1 << bF + eps;

    dw->coeffRef(0,0) = (cost(m1, bias, data) - c)/eps;
    dw->coeffRef(0,1) = (cost(m2, bias, data) - c)/eps;
    b->coeffRef(0,0) = (cost(weights, b1, data) - c)/eps;
}

int main1()
{
    Eigen::Matrix<float, 1, 2> weights;
    Eigen::Matrix<float, 1, 1> bias;

    weights.setRandom();
    bias.setRandom();

    TrainingData or_gate;
    or_gate.activations[0] << 0, 0;
    or_gate.activations[1] << 1, 0;
    or_gate.activations[2] << 0, 1;
    or_gate.activations[3] << 1, 1;
    or_gate.results = { 0, 1, 1, 1 };

    TrainingData and_gate;
    and_gate.activations[0] << 0, 0;
    and_gate.activations[1] << 1, 0;
    and_gate.activations[2] << 0, 1;
    and_gate.activations[3] << 1, 1;
    and_gate.results = { 0, 0, 0, 1 };

    TrainingData nand_gate;
    nand_gate.activations[0] << 0, 0;
    nand_gate.activations[1] << 1, 0;
    nand_gate.activations[2] << 0, 1;
    nand_gate.activations[3] << 1, 1;
    nand_gate.results = { 1, 1, 1, 0 };
    
    TrainingData xor_gate;
    xor_gate.activations[0] << 0, 0;
    xor_gate.activations[1] << 1, 0;
    xor_gate.activations[2] << 0, 1;
    xor_gate.activations[3] << 1, 1;
    xor_gate.results = { 0, 1, 1, 0 };

    TrainingData data = or_gate;

    // STARTING

    std::cout << "Weights: " << weights <<  std::endl;
    std::cout << "Biases: " << bias << std::endl;
    for (int i = 0; i < 4; i++)
    {
        auto result = (weights * data.activations[i]) + bias;
        auto sigM = result.unaryExpr(&sigmoidf);
        std::cout << "Result " << i << ": " << sigM << std::endl;
    }
    std::cout << "Cost: " << cost(weights, bias, data) << std::endl;

    float eps = 1e-1;
    float rate = 1e-1;

    for (int i = 0; i < 10*10000; i++)
    {
        float c = cost(weights, bias, data);
        wMatrix m;
        bMatrix b;

        derivative_cost(weights, bias, eps, data, &m, &b);

        weights.coeffRef(0,0) -= rate*(m.coeff(0,0));
        weights.coeffRef(0,1) -= rate*(m.coeff(0,1));
        bias.coeffRef(0,0) -= rate*(b.coeff(0,0));
    }    

    std::cout << "Weights: " << weights <<  std::endl;
    std::cout << "Biases: " << bias << std::endl;
    for (int i = 0; i < 4; i++)
    {
        auto result = (weights * data.activations[i]) + bias;
        auto sigM = result.unaryExpr(&sigmoidf);
        std::cout << "Result " << i << ": " << sigM << std::endl;
    }
    std::cout << "Cost: " << cost(weights, bias, data) << std::endl;
}

int main()
{
    int layout[2] = { 2, 1 };
    NN::TrainingData or_gate(4, layout);
    or_gate.inputs[0] << 0, 0;
    or_gate.inputs[1] << 1, 0;
    or_gate.inputs[2] << 0, 1;
    or_gate.inputs[3] << 1, 1;
    or_gate.outputs[0] << 0;
    or_gate.outputs[1] << 1;
    or_gate.outputs[2] << 1;
    or_gate.outputs[3] << 1;

    NN::TrainingData and_gate(4, layout);
    and_gate.inputs[0] << 0, 0;
    and_gate.inputs[1] << 1, 0;
    and_gate.inputs[2] << 0, 1;
    and_gate.inputs[3] << 1, 1;
    and_gate.outputs[0] << 0;
    and_gate.outputs[1] << 0;
    and_gate.outputs[2] << 0;
    and_gate.outputs[3] << 1;

    NN::TrainingData nand_gate(4, layout);
    nand_gate.inputs[0] << 0, 0;
    nand_gate.inputs[1] << 1, 0;
    nand_gate.inputs[2] << 0, 1;
    nand_gate.inputs[3] << 1, 1;
    nand_gate.outputs[0] << 1;
    nand_gate.outputs[1] << 1;
    nand_gate.outputs[2] << 1;
    nand_gate.outputs[3] << 0;
    
    /*
    
    NN::TrainingData xor_gate(4, layout); // WILL NOT WORK
    xor_gate.inputs[0] << 0, 0;
    xor_gate.inputs[1] << 1, 0;
    xor_gate.inputs[2] << 0, 1;
    xor_gate.inputs[3] << 1, 1;
    xor_gate.outputs = { 0, 1, 1, 0 };

    */

   NN::BaseNetwork nn(layout, 2, nand_gate, 100*1000);
   nn.Learn(1e-1, 1e-1);
   nn.PrintResults();
}