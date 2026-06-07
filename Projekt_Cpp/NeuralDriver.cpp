#include "NeuralDriver.h"

#include <array>
#include <cmath>
#include <iostream>

#include "AIInputBuilder.h"

static float Sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

NeuralDriver::NeuralDriver(const std::wstring& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "NeuralDriver"),
    sessionOptions(),
    session(nullptr)
{
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session = Ort::Session(env, modelPath.c_str(), sessionOptions);

    auto inputNameAllocated = session.GetInputNameAllocated(0, allocator);
    inputName = inputNameAllocated.get();

    auto outputNameAllocated = session.GetOutputNameAllocated(0, allocator);
    outputName = outputNameAllocated.get();

    std::cout << "ONNX model loaded.\n";
    std::cout << "Input name: " << inputName << "\n";
    std::cout << "Output name: " << outputName << "\n";
}

std::vector<float> NeuralDriver::PredictProbabilities(const GameState& state)
{
    std::vector<float> input = BuildAIInputFromGameState(state);

    if (input.size() != 81)
    {
        std::cout << "Wrong AI input size: " << input.size() << "\n";
        return { 0.0f, 0.0f, 0.0f, 0.0f };
    }

    std::array<int64_t, 2> inputShape = {
        1,
        static_cast<int64_t>(input.size())
    };

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        input.data(),
        input.size(),
        inputShape.data(),
        inputShape.size()
    );

    const char* inputNames[] = { inputName.c_str() };
    const char* outputNames[] = { outputName.c_str() };

    auto outputTensors = session.Run(
        Ort::RunOptions{ nullptr },
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );

    float* outputData = outputTensors[0].GetTensorMutableData<float>();

    // Model zwraca logity, bo w Pythonie trenowaliśmy z BCEWithLogitsLoss.
    // Dlatego tutaj robimy sigmoid.
    std::vector<float> probabilities(4);

    for (int i = 0; i < 4; i++)
    {
        probabilities[i] = Sigmoid(outputData[i]);
    }

    return probabilities;
}

Controls NeuralDriver::PredictControls(const GameState& state)
{
    std::vector<float> probs = PredictProbabilities(state);

    constexpr float ACCELERATE_THRESHOLD = 0.57f;
    constexpr float BRAKE_THRESHOLD = 0.59f;
    constexpr float STEER_LEFT_THRESHOLD = 0.575f;
    constexpr float STEER_RIGHT_THRESHOLD = 0.465f;

    float pAccelerate = probs[0];
    float pBrake = probs[1];
    float pSteerLeft = probs[2];
    float pSteerRight = probs[3];

    Controls controls{};
    controls.accelerate = 0.0f;
    controls.brake = 0.0f;
    controls.steerLeft = 0.0f;
    controls.steerRight = 0.0f;

    // Gaz / hamulec.
    // Hamulec ma priorytet, żeby auto nie wciskało gazu i hamulca naraz.
    if (pBrake > BRAKE_THRESHOLD)
    {
        controls.brake = 1.0f;
    }
    else if (pAccelerate > ACCELERATE_THRESHOLD)
    {
        controls.accelerate = 1.0f;
    }

    // Skręt.
    bool wantsLeft = pSteerLeft > STEER_LEFT_THRESHOLD;
    bool wantsRight = pSteerRight > STEER_RIGHT_THRESHOLD;

    if (wantsLeft || wantsRight)
    {
        if (wantsLeft && wantsRight)
        {
            // Gdy oba kierunki przekroczą próg, wybierz pewniejszy.
            if (pSteerLeft > pSteerRight)
            {
                controls.steerLeft = 1.0f;
            }
            else
            {
                controls.steerRight = 1.0f;
            }
        }
        else if (wantsLeft)
        {
            controls.steerLeft = 1.0f;
        }
        else
        {
            controls.steerRight = 1.0f;
        }
    }

    return controls;
}