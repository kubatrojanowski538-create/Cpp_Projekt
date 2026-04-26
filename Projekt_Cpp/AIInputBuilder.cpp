#include "AIInputBuilder.h"

#include <iostream>
#include "AIScalerData.h"

static float ScaleNumericInput(float value, int index)
{
    return (value - AI_INPUT_MEAN[index]) / AI_INPUT_SCALE[index];
}

std::vector<float> BuildAIInputFromGameState(const GameState& state)
{
    std::vector<float> input;
    input.reserve(81);

    // 1. car_speed
    input.push_back(ScaleNumericInput(state.speed, 0));

    // 2. ray_dist_0 ... ray_dist_19
    for (int i = 0; i < 20; i++)
    {
        input.push_back(ScaleNumericInput(state.rayDistances[i], i + 1));
    }

    // 3. ray_type one-hot
    // Python robil kolejnosc: -1, 0, 1
    for (int i = 0; i < 20; i++)
    {
        int rayType = state.rayTypes[i];

        if (rayType == -1)
        {
            input.push_back(1.0f);
            input.push_back(0.0f);
            input.push_back(0.0f);
        }
        else if (rayType == 0)
        {
            input.push_back(0.0f);
            input.push_back(1.0f);
            input.push_back(0.0f);
        }
        else if (rayType == 1)
        {
            input.push_back(0.0f);
            input.push_back(0.0f);
            input.push_back(1.0f);
        }
        else
        {
            // Awaryjnie: nieznany typ traktujemy jak brak trafienia.
            input.push_back(1.0f);
            input.push_back(0.0f);
            input.push_back(0.0f);
        }
    }

    return input;
}

void PrintAIInputDebug(const std::vector<float>& input)
{
    std::cout << "AI input size: " << input.size() << "\n";

    for (int i = 0; i < input.size(); i++)
    {
        std::cout << i << ": " << input[i] << "\n";
    }
}