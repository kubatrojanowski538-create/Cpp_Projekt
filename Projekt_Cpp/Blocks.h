#pragma once
#include "Drawable.h"
#include <iostream>
class Blocks :
    public Drawable
{
public:
    int inFileType;

    virtual int getBlockType() { return 0; }

    virtual bool checkCollision(Vector2 point) { return false; }
    bool shouldDraw(float drawSize);
    virtual void drawBlock();
    virtual void scaleBlock();
    virtual void readBlock(std::istream& in);
    virtual void writeBlock(std::ostream& out);


};

