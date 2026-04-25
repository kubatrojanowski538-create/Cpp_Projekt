#pragma once
#include "Util.h"
#include <string>
#include <fstream>



void SaveGameStateHeader(const std::string& fileName);
void AppendGameStateToFile(const GameState& state, const std::string& fileName);
