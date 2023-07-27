#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <array>

#include "NeuralNetwork.hpp"

int main()
{
    int layout[3] = { 2, 2, 1 };
    NN::ActF actFs[] = { NN::ActF::SIGMOID, NN::ActF::SIGMOID };

    NN::TrainingData or_gate(4, layout, 2);
    or_gate.inputs[0] << 0, 0;
    or_gate.inputs[1] << 1, 0;
    or_gate.inputs[2] << 0, 1;
    or_gate.inputs[3] << 1, 1;
    or_gate.outputs[0] << 0;
    or_gate.outputs[1] << 1;
    or_gate.outputs[2] << 1;
    or_gate.outputs[3] << 1;

    NN::TrainingData and_gate(4, layout, 2);
    and_gate.inputs[0] << 0, 0;
    and_gate.inputs[1] << 1, 0;
    and_gate.inputs[2] << 0, 1;
    and_gate.inputs[3] << 1, 1;
    and_gate.outputs[0] << 0;
    and_gate.outputs[1] << 0;
    and_gate.outputs[2] << 0;
    and_gate.outputs[3] << 1;

    NN::TrainingData nand_gate(4, layout, 2);
    nand_gate.inputs[0] << 0, 0;
    nand_gate.inputs[1] << 1, 0;
    nand_gate.inputs[2] << 0, 1;
    nand_gate.inputs[3] << 1, 1;
    nand_gate.outputs[0] << 1;
    nand_gate.outputs[1] << 1;
    nand_gate.outputs[2] << 1;
    nand_gate.outputs[3] << 0;
    
    NN::TrainingData xor_gate(4, layout, 2);
    xor_gate.inputs[0] << 0, 0;
    xor_gate.inputs[1] << 1, 0;
    xor_gate.inputs[2] << 0, 1;
    xor_gate.inputs[3] << 1, 1;
    xor_gate.outputs[0] << 0;
    xor_gate.outputs[1] << 1;
    xor_gate.outputs[2] << 1;
    xor_gate.outputs[3] << 0;

   NN::MultiLayeredNetwork or_nn(layout, 3, or_gate, actFs, 10*1000, NN::CostF::MSE);
   or_nn.Learn(1e-1, 1e-1);
   or_nn.PrintResults();

   NN::MultiLayeredNetwork xor_nn(layout, 3, xor_gate, actFs, 100*1000, NN::CostF::MSE);
   xor_nn.Learn(1e-1, 1e-1);
   xor_nn.PrintResults();
}