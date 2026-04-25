#pragma once
#include "GameState.h"



void SaveGameStateHeader(const std::string& fileName)
{
	std::ofstream file(fileName);

	file << "car_speed,";
	file << "accelerate,brake,steer_left,steer_right";

	for (int i = 0; i < 20; i++) {
		file << ",ray_dist_" << i;
		file << ",ray_type_" << i;
	}

	file << "\n";
}

void AppendGameStateToFile(const GameState& state, const std::string& fileName)
{
	std::ofstream file(fileName, std::ios::app);

	
	
	file << state.speed << ",";

	file << state.inputs.accelerate << ",";
	file << state.inputs.brake << ",";
	file << state.inputs.steerLeft << ",";
	file << state.inputs.steerRight;

	for (int i = 0; i < 20; i++) {
		file << ",";
		file << state.rayDistances[i];
		file << ",";
		file << state.rayTypes[i];
	}

	file << "\n";
}
