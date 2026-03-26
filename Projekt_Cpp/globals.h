#pragma once
#include <vector>
#include "raylib.h"
#include "Blocks.h"

extern const int windowWidth;
extern const int windowHeight;
extern float camOffsetX;
extern float camOffsetY;
extern const int fps;
extern std::vector<Blocks*> klocki;
extern int drawScale;
extern bool isDrawing;
extern Color backgroundColor;

extern Vector2 respawnPoint; 
extern float gameTime;       
extern bool gameFinished;    
extern bool timerRunning;    