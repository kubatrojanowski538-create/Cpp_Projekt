#pragma once
#include "Drawable.h"
#include "raylib.h"

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

public:
    void drawCar();
    void resetCar();
    void updateSpeed();
    void updatePosition();
    void rotateCar();
    Car();
    




};

