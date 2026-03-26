#pragma once
#include "Blocks.h"
#include "Util.h"
class BarrierLine :
    public Blocks
    
{
public:
    float length;
    Vector2 start, end;

    
    BarrierLine();
    BarrierLine(int x);
    BarrierLine(float len, float rot,float posX, float posY, Vector2 start, Vector2 end, int fileType );
    void drawBlock() override;
    void scaleBlock() override;
    int getBlockType() override;
    bool checkCollision(Vector2 point) override;
    void readBlock(std::istream& in) override;
    void writeBlock(std::ostream& out) override;


};

