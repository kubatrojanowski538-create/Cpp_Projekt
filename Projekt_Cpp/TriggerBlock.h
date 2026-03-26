#pragma once
#include "Blocks.h"
#include "raylib.h"

class TriggerBlock : public Blocks
{
public:
    int type; 
    float width, height;

    TriggerBlock(); 
    TriggerBlock(int x);
    void drawBlock() override;
    void scaleBlock() override;
    bool checkCollision(Vector2 point) override;
    int getBlockType() override;
    void readBlock(std::istream& in) override;
    void writeBlock(std::ostream& out) override;
};