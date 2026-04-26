#pragma once

#include <vector>
#include "Util.h"

std::vector<float> BuildAIInputFromGameState(const GameState& state);
void PrintAIInputDebug(const std::vector<float>& input);
