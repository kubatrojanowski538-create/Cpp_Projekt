#pragma once
#include "GameState.h"



void SaveGameStateHeader(const std::string& fileName)
{
	std::ofstream file(fileName);

	file << "car_speed";

	for (int i = 0; i < 20; i++) {
		file << ",ray_dist_" << i;
		file << ",ray_type_" << i;
	}
	file << ",accelerate,brake,steer_left,steer_right";

	

	file << "\n";
}

void AppendGameStateToFile(const GameState& state, const std::string& fileName)
{
	std::ofstream file(fileName, std::ios::app);

	if (!file.is_open()) {
		std::cout << "Nie udalo sie otworzyc pliku do zapisu: " << fileName << "\n";
		return;
	}

	file << state.speed;

	for (int i = 0; i < 20; i++) {
		file << ",";
		file << state.rayDistances[i];
		file << ",";
		file << state.rayTypes[i];
	}

	file << "," << state.inputs.accelerate;
	file << "," << state.inputs.brake;
	file << "," << state.inputs.steerLeft;
	file << "," << state.inputs.steerRight;

	file << "\n";

	if (file.fail()) {
		std::cout << "Blad podczas zapisu do pliku: " << fileName << "\n";
	}
}

std::string EnsureGameStateFileExists(const std::string& fileName)
{
	std::ifstream file(fileName, std::ios::ate);

	bool fileExists = file.good();
	bool fileIsEmpty = true;

	if (fileExists)
	{
		fileIsEmpty = file.tellg() == 0;
	}

	file.close();

	if (!fileExists || fileIsEmpty)
	{
		SaveGameStateHeader(fileName);
	}

	return fileName;
}

