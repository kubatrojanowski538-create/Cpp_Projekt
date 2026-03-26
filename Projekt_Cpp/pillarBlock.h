#pragma once
#include "Blocks.h"
class pillarBlock :
    public Blocks
{
public:

    float radius;
    pillarBlock();
    pillarBlock(int x);
    void drawBlock() override;
    void scaleBlock() override;
    int getBlockType() override;
    bool checkCollision(Vector2 point) override;
    void readBlock(std::istream& in) override;
    void writeBlock(std::ostream& out) override;

};

