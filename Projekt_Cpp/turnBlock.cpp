#include "turnBlock.h"
#include "raylib.h"
#include "globals.h"
#include <iostream>
#include "raymath.h"
using namespace std;
turnBlock::turnBlock(){

}

turnBlock::turnBlock(int x)
{

}

turnBlock::turnBlock(Vector2 points[91], int spread, int filetype)
{
	this->points.resize(91);
	for (int i = 0; i < 91; i++) {
		this->points[i] = points[i];
	}
	this->spread = spread;
	this->inFileType = filetype;
}


void turnBlock::drawBlock()
{
	
	
	for (int i = 0; i < this->spread * 2; i++) {
		DrawLineEx(
			{ this->points[i].x - camOffsetX , this->points[i].y - camOffsetY},
			{ this->points[i + 1].x - camOffsetX, this->points[i + 1].y - camOffsetY },
			16, WHITE);
	}
	
	
}

void turnBlock::scaleBlock()
{
	for (int i = 0; i < this->points.size(); i++) {
		this->points[i].x *= drawScale;
		this->points[i].y *= drawScale;
	}
}
bool turnBlock::checkCollision(Vector2 point) {
	for (size_t i = 0; i < this->spread*2; i++) {
		Vector2 a = this->points[i];
		Vector2 b = this->points[i + 1];

		float l2 = pow(a.x - b.x, 2) + pow(a.y - b.y, 2);
		if (l2 == 0) continue;

		float t = ((point.x - a.x) * (b.x - a.x) + (point.y - a.y) * (b.y - a.y)) / l2;
		t = fmax(0, fmin(1, t));

		Vector2 projection = { a.x + t * (b.x - a.x), a.y + t * (b.y - a.y) };
		float dist = sqrt(pow(point.x - projection.x, 2) + pow(point.y - projection.y, 2));

		if (dist < 8.0f) return true;
	}
	return false;
}
void turnBlock::readBlock(std::istream& in)
{
	this->points.resize(91);
	for (int i = 0; i < 91; i++) {
		in >> points[i];
	}
	in >> this->spread;

}
void turnBlock::writeBlock(std::ostream& out)
{
	out << this->inFileType << " ";
	for (Vector2 point : this->points) {
		out << point;
	}
	out << this->spread << endl;

}
int turnBlock::getBlockType() { return 0; } 
