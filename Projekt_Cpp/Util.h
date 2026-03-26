#pragma once

#include "raylib.h"
#include <string>
#include "globals.h"
#include <iostream>

std::string GetText();
std::ostream& operator<<(std::ostream& out, Vector2 Vec);
std::istream& operator>>(std::istream& in, Vector2& Vec);

