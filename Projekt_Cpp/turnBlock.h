#pragma once
#include "Blocks.h"
#include <vector>
#include "Util.h"
class turnBlock :
    public Blocks
{
public:
    float radius;
    float spread;
    Vector2 A, B, C, circleOrigin;
    std::vector<Vector2> points;


    turnBlock();
    turnBlock(int x);
    turnBlock(Vector2 points[91], int spread, int filetype);
    void drawBlock() override;
    void scaleBlock() override;
    int getBlockType() override;
    bool checkCollision(Vector2 point) override;
    void readBlock(std::istream& in) override;
    void writeBlock(std::ostream& out) override;
};


