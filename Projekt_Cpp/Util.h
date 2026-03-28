#pragma once

#include "raylib.h"
#include <string>
#include "globals.h"
#include <iostream>
#include "BarrierLine.h"
#include "pillarBlock.h"
#include "turnBlock.h"
#include "TriggerBlock.h"
#include "math.h"

struct Controls
{
	float steerRight;
	float steerLeft;
	float accelerate;
	float brake;
};

std::string GetText();
std::ostream& operator<<(std::ostream& out, Vector2 Vec);
std::istream& operator>>(std::istream& in, Vector2& Vec);
Blocks* SetupBlock(int type);
Controls GetInputs();
Vector2 NormalizeVector2(Vector2 vec);


float RayDistance2D(Vector2 P, Vector2 D, Vector2 A, Vector2 B, float& outDist);
float cross(Vector2 a, Vector2 b);



