#include "Car.h"
#include "raylib.h"
#include "math.h"
#include <iostream>
#include "globals.h"
#include "raymath.h"

using namespace std;

Car::Car() {
    this->posX = 0; 
    this->posY = 0;
    this->rotation = 0;
    this->scale = 0.125;
    this->speed = 0;
    this->velX = 0;
    this->velY = 0;
    this->respawnRot = 0;
    this->image = LoadImage("assets/car.png");
    this->carTexture = LoadTextureFromImage(this->image);
    this->drawRect = { 0, 0 , float(this->image.width), float(this->image.height) };
}

void Car::drawCar() {
    
    
    
    Vector2 origin = { float(this->image.width * this->scale) / 2, float(this->image.height * this->scale) / 2 };
    DrawTexturePro(this->carTexture, this->drawRect, this->carRect, origin, this->rotation + 270, WHITE);
    
}

void Car::resetCar() {
    this->speed = 0;
    this->rotation = respawnRot;
    if (respawnPoint.x != 0 && respawnPoint.y != 0) {
        this->posX = respawnPoint.x;
        this->posY = respawnPoint.y;
    }
    else {
        bool foundStart = false;
        for (Blocks* b : klocki) {
            if (b->getBlockType() == 1) {
                this->posX = b->posX;
                this->posY = b->posY;
                foundStart = true;
                break;
            }
        }
        gameTime = 0;
        timerRunning = false;
        gameFinished = false;
    }

    camOffsetX = this->posX - windowWidth / 2;
    camOffsetY = this->posY - windowHeight / 2;
}

void Car::updateSpeedRot(Controls inputs) {
    //speed
    this->speed *= 0.995f;
    if (inputs.accelerate && (this->speed < 20)) {
        this->speed += 0.3f;
        if (!timerRunning && !gameFinished) timerRunning = true; 
    }
    if (inputs.brake && (this->speed > -5)) {
        this->speed -= 0.3f;
    }

    if (this->speed < -5) this->speed = -5;
    if (this->speed > 20) this->speed = 20;

    this->velX = cos(this->rotation * DEG2RAD) * this->speed;
    this->velY = sin(this->rotation * DEG2RAD) * this->speed;

    //rot
    float turnSpeed = 0;
    float maxTurn = 4.0f;

    if (abs(this->speed) < 0.5f) {
        turnSpeed = 0;
    }
    else if (abs(this->speed) <= 10.0f) {
        turnSpeed = (this->speed / 10.0f) * maxTurn;
    }
    else {

        float factor = 1.0f - 0.5f * ((abs(this->speed) - 10.0f) / 10.0f);
        turnSpeed = maxTurn * factor;
    }

    if (inputs.steerLeft) this->rotation -= turnSpeed;
    if (inputs.steerRight) this->rotation += turnSpeed;
}

void Car::updatePosition() {
    this->posX += velX;
    this->posY += velY;

    camOffsetX = this->posX - windowWidth / 2;
    camOffsetY = this->posY - windowHeight / 2;


    this->carRect.x = (this->posX - camOffsetX) - (this->carRect.width / 2);
    this->carRect.y = (this->posY - camOffsetY) - (this->carRect.height / 2);

    this->carRect.width = this->image.width * this->scale;
    this->carRect.height = this->image.height * this->scale;
}



void Car::updateCar(Controls inputs)
{
    this->updateSpeedRot(inputs);
    this->updatePosition();
    this->checkCollision();
    this->UpdateRays();

}

void Car::checkCollision()
{
    Vector2 imagecentre = { windowWidth / 2 - this->image.width / 16, windowHeight / 2 - this->image.height / 16 };

    float temprot1 = this->rotation - 26;
    float temprot2 = this->rotation + 26;
    float temprot3 = this->rotation + 90;

    Vector2 points[6];
    points[0].x = imagecentre.x + cos(temprot1 * DEG2RAD) * 50;
    points[0].y = imagecentre.y + sin(temprot1 * DEG2RAD) * 50;

    points[1].x = imagecentre.x - cos(temprot1 * DEG2RAD) * 50;
    points[1].y = imagecentre.y - sin(temprot1 * DEG2RAD) * 50;

    points[2].x = imagecentre.x - cos(temprot2 * DEG2RAD) * 50;
    points[2].y = imagecentre.y - sin(temprot2 * DEG2RAD) * 50;

    points[3].x = imagecentre.x + cos(temprot2 * DEG2RAD) * 50;
    points[3].y = imagecentre.y + sin(temprot2 * DEG2RAD) * 50;

    points[4].x = imagecentre.x - cos(temprot3 * DEG2RAD) * 25;
    points[4].y = imagecentre.y - sin(temprot3 * DEG2RAD) * 25;

    points[5].x = imagecentre.x + cos(temprot3 * DEG2RAD) * 25;
    points[5].y = imagecentre.y + sin(temprot3 * DEG2RAD) * 25;


    for (Vector2 punkt : points) {
        punkt = Vector2Add(punkt, { camOffsetX, camOffsetY });
        for (Blocks* klocek : klocki) {
            if (klocek->checkCollision(punkt)) {
                int type = klocek->getBlockType();
                if (type == 0) {
                    this->resetCar();
                }
                if (type == 1) {
                }
                if (type == 2) {
                    respawnPoint = { imagecentre.x + camOffsetX + 25, imagecentre.y + camOffsetY + 50 };
                    this->respawnRot = this->rotation;
                }
                if (type == 3) {
                    if (!gameFinished) {
                        gameFinished = true;
                        timerRunning = false;
                    }
                }
            }
        }
    }
}

void Car::UpdateRays()
{
    for (int i = 0; i < 13; i++) {
        this->Rays[i] = {
            cos((-60 + 10 * i + this->rotation) * DEG2RAD),
            sin((-60 + 10 * i + this->rotation) * DEG2RAD)
        };
        
    }
    for (int i = 0; i < 7; i++) {
        this->Rays[i + 13] = {
            cos((100 + 30 * i + this->rotation) * DEG2RAD),
            sin((100 + 30 * i + this->rotation) * DEG2RAD)
        };
        

    }
    
}
