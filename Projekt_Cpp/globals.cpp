#include "globals.h"

const int windowWidth = 1620;
const int windowHeight = 1080;
float camOffsetX = 0;
float camOffsetY = 0;
const int fps = 60;
std::vector<Blocks*> klocki;
int drawScale = 1;
bool isDrawing = true;
Color backgroundColor = { 0, 100, 55, 255 };

Vector2 respawnPoint = { 0, 0 };
float gameTime = 0.0f;
bool gameFinished = false;
bool timerRunning = false;
float maxRayDistance = 1000.0f;

/*
Line = 0
Pillar = 1
Turn = 2
Start = 3
Check = 4
Fin = 5





*/