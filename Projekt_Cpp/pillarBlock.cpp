#include "pillarBlock.h"
#include "globals.h"

pillarBlock::pillarBlock(){

}

pillarBlock::pillarBlock(int x)
{

}

pillarBlock::pillarBlock(float posX, float posY, float rad, int fileType)
{
	this->posX = posX;
	this->posY = posY;
	this->radius = rad;
	this->inFileType = fileType;

}

void pillarBlock::drawBlock()
{	
	float posx = this->posX - camOffsetX;
	float posy = this->posY - camOffsetY;
	float radius = this->radius ;

	DrawCircle(posx, posy, radius, WHITE);
}

void pillarBlock::scaleBlock()
{
	this->posX *= drawScale;
	this->posY *= drawScale;
	this->radius *= drawScale;
}
bool pillarBlock::checkCollision(Vector2 point) {
	float dx = point.x - this->posX;
	float dy = point.y - this->posY;
	float dist = sqrt(dx * dx + dy * dy);
	return dist < this->radius;
}
void pillarBlock::readBlock(std::istream& in)
{
	in >> this->posX >> this->posY >> this->radius;
}
void pillarBlock::writeBlock(std::ostream& out)
{
	out << this->inFileType << " " << this->posX << " " << this->posY << " " << this->radius << std::endl;
}
int pillarBlock::getBlockType() { return 0; } 
