#pragma once
#include "Util.h"
#include <string>
#include <fstream>
#include "Car.h"



void SaveGameStateHeader(const std::string& fileName);
void AppendGameStateToFile(const GameState& state, const std::string& fileName);
RayAndType GetClosestRayHit(Vector2 position, Vector2 rayDirection);
