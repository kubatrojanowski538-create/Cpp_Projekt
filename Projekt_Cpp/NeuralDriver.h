#pragma once

#include <string>
#include <vector>

#include "Util.h"
#include "onnxruntime_cxx_api.h"

class NeuralDriver
{
public:
    NeuralDriver(const std::wstring& modelPath);

    Controls PredictControls(const GameState& state);
    std::vector<float> PredictProbabilities(const GameState& state);

private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string inputName;
    std::string outputName;
};