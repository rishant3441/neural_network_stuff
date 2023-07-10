#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <array>

#include "NeuralNetwork.hpp"

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