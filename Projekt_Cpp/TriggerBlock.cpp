#include "TriggerBlock.h"
#include "globals.h"
#include "raylib.h"
#include "raymath.h"

TriggerBlock::TriggerBlock() {
}

TriggerBlock::TriggerBlock(int x)
{
}

TriggerBlock::TriggerBlock(float posX, float posY, float width, float height, int type, int fileType)
{
    this->posX = posX;
    this->posY = posY;
    this->width = width;
    this->height = height;
    this->type = type;
    this->inFileType = fileType;

}

void TriggerBlock::drawBlock() {
    Color c = WHITE;
    if (this->type == 1) c = { 0, 255, 0, 255 }; 
    if (this->type == 2) c = { 0, 0, 255, 255 }; 
    if (this->type == 3) c = { 255, 0, 0, 255 }; 

    DrawRectangle(
        (this->posX - width / 2) - camOffsetX,
        (this->posY - height / 2) - camOffsetY,
        width, height, c
    );
}

void TriggerBlock::scaleBlock() {
    this->posX *= drawScale;
    this->posY *= drawScale;
    this->width *= drawScale;
    this->height *= drawScale;
}

bool TriggerBlock::checkCollision(Vector2 point) {
    float left = this->posX - width / 2;
    float right = this->posX + width / 2;
    float top = this->posY - height / 2;
    float bottom = this->posY + height / 2;

    return (point.x >= left && point.x <= right && point.y >= top && point.y <= bottom);
}

int TriggerBlock::getBlockType() {
    return this->type;
}

void TriggerBlock::readBlock(std::istream& in)
{
    in >> this->type >> this->posX >> this->posY >> this->width >> this->height;
}

void TriggerBlock::writeBlock(std::ostream& out)
{
    out << this->inFileType << " " << this->type << " ";
    out << this->posX << " " << this->posY << " ";
    out << this->width << " " << this->height << std::endl;
}
