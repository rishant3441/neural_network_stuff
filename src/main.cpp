#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <array>

#include <omp.h>
#include "NeuralNetwork.hpp"

int main()
{
    omp_set_dynamic(0);
    omp_set_num_threads(4);

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

   NN::MultiLayeredNetwork or_nn(layout, 3, actFs, NN::CostF::MSE);
   or_nn.Learn(or_gate, 10*1000, 1e-1, 1e-1);
   or_nn.PrintResults();

   NN::MultiLayeredNetwork xor_nn(layout, 3, actFs, NN::CostF::MSE);
   xor_nn.Learn(xor_gate, 100*1000, 1e-1, 1e-1);
   xor_nn.Save("./xor_nn.json");

   NN::MultiLayeredNetwork test_nn(layout, 3, actFs, NN::CostF::MSE);

   test_nn.Load("./xor_nn.json");
   test_nn.Learn(xor_gate, 3, 1e-1, 1e-1);
   test_nn.PrintResults();
}