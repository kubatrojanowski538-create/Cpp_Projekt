#include "Blocks.h"
#include "globals.h"

bool Blocks::shouldDraw(float drawSize)
{

    
    if (isDrawing) {
        return 1;
    }

    float relativePosX = (this->posX * drawScale) - camOffsetX;
    float relativePosY = (this->posY * drawScale) - camOffsetY;

    if ((relativePosX < -1 * drawSize ) || relativePosX > (1280 + drawSize)) return 0;
    if ((relativePosY < -1 * drawSize ) || relativePosY > (720 + drawSize)) return 0;
    return 1;
    
} 
void Blocks::drawBlock() {

}
void Blocks::scaleBlock() {

}

void Blocks::readBlock(std::istream& in)
{
}

void Blocks::writeBlock(std::ostream& out)
{
}





 
 
 
 
 
 
 
 
 
 
 
 



