#include "BarrierLine.h"
#include "raylib.h"
#include "math.h"
#include <iostream>
#include "globals.h"
#include "raymath.h"
using namespace std;

BarrierLine::BarrierLine() {	

}

BarrierLine::BarrierLine(int x)
{

}

BarrierLine::BarrierLine(float len, float rot, float posX, float posY, Vector2 start, Vector2 end, int fileType)
{
	this->length = len;
	this->rotation = rot;
	this->posX = posX;
	this->posY = posY;
	this->start = start;
	this->end = end;
	this->inFileType = fileType;

}

void BarrierLine::drawBlock() {
	
		
	Vector2 drawStart = { this->start.x - camOffsetX , this->start.y - camOffsetY };
	Vector2 drawEnd = { this->end.x - camOffsetX, this->end.y - camOffsetY };
	DrawLineEx(drawStart, drawEnd, 16, WHITE);

}

void BarrierLine::scaleBlock()
{	
	this->start = Vector2Scale(this->start, drawScale);
	this->end = Vector2Scale(this->end, drawScale);
}

bool BarrierLine::checkCollision(Vector2 point) {

	Vector2 p = point;
	Vector2 a = this->start;
	Vector2 b = this->end;

	float l2 = pow(a.x - b.x, 2) + pow(a.y - b.y, 2);
	if (l2 == 0) return false;

	float t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2;
	t = fmax(0, fmin(1, t));

	Vector2 projection = { a.x + t * (b.x - a.x), a.y + t * (b.y - a.y) };

	float dist = sqrt(pow(p.x - projection.x, 2) + pow(p.y - projection.y, 2));

	return dist < 8.0f;
}
void BarrierLine::readBlock(std::istream& in)
{
	in >> this->posX >> this->posY;
	in >> this->start >> this->end;

}
void BarrierLine::writeBlock(std::ostream& out)
{
	out << this->inFileType << " ";
	out << this->posX << " " << this->posY << " ";
	out << this->start  <<  this->end << endl;

}
int BarrierLine::getBlockType() { return 0; } 
