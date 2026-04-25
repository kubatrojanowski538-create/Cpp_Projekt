#pragma once
#include "Drawable.h"
#include "raylib.h"
#include "Util.h"

class Car :
    public Drawable
{
public:

        float speed, velX, velY;
        float respawnRot;
        Rectangle drawRect;
        Rectangle carRect;
        Image image;
        Texture carTexture;
        Vector2 Rays[20];

public:
    void drawCar();
    void resetCar();
    void updateSpeedRot(Controls inputs);
    void updatePosition();
    void updateCar(Controls inputs);
    void checkCollision();
    void UpdateRays();
    Car();
    void VisualiseRays();




};

