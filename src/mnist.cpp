#include "NeuralNetwork.hpp"

#include <csv.h>
#include <string>
#include <sstream>
#include <iostream>

void parseComma(const std::string& string, std::vector<float>* vector)
{
    std::stringstream ss(string);

    for (int i; ss >> i;) {
        vector->push_back(i);    
        if (ss.peek() == ',')
            ss.ignore();
    }    
}

void parseData(NN::TrainingData* data, const std::string& file, int index)
{
    io::LineReader reader(file);
    std::vector<float> line;
    line.reserve(785);
    int num;
    int ran = 0;
    while (char* in = reader.next_line())
    {
        if (ran == index)
        {
            break;
        }

        parseComma(in, &line);

        for (int i = 0; i < line.size(); i++)
        {
            if (i == 0)
            {
                continue;
            }
            line[i] /= 255;
        }

        num = line[0];
        line.erase(line.begin());

        data->inputs[ran] = Eigen::Map<Eigen::Matrix<float, 784, 1>>(line.data());
        if (num == 0)
        {
            data->outputs[ran] << 1,0,0,0,0,0,0,0,0,0; 
        }
        if (num == 1)
        {
            data->outputs[ran] << 0,1,0,0,0,0,0,0,0,0; 
        }
        if (num == 2)
        {
            data->outputs[ran] << 0,0,1,0,0,0,0,0,0,0; 
        }
        if (num == 3)
        {
            data->outputs[ran] << 0,0,0,1,0,0,0,0,0,0; 
        }
        if (num == 4)
        {
            data->outputs[ran] << 0,0,0,0,1,0,0,0,0,0; 
        }
        if (num == 5)
        {
            data->outputs[ran] << 0,0,0,0,0,1,0,0,0,0; 
        }
        if (num == 6)
        {
            data->outputs[ran] << 0,0,0,0,0,0,1,0,0,0; 
        }
        if (num == 7)
        {
            data->outputs[ran] << 0,0,0,0,0,0,0,1,0,0; 
        }
        if (num == 8)
        {
            data->outputs[ran] << 0,0,0,0,0,0,0,0,1,0; 
        }
        if (num == 9)
        {
            data->outputs[ran] << 0,0,0,0,0,0,0,0,0,1; 
        }

        line.clear();
        line.reserve(785);
        ran++;
    }
}

int main()
{
    int layout[] = { 784, 16, 16, 10 };
    NN::ActF activationFs[] = { NN::ActF::RELU, NN::ActF::RELU, NN::ActF::SOFTMAX };
    NN::TrainingData data(100, layout, 3);

    parseData(&data, "mnist_train.csv", 100);     

    NN::MultiLayeredNetwork network(layout, 4, data, activationFs, 500, NN::CostF::CROSS_ENTROPY);
    network.PrintResults();
    network.Learn();
    network.PrintResults();
    
    std::cin.get();
}