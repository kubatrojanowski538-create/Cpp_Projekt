#include "turnBlock.h"
#include "raylib.h"
#include "globals.h"
#include <iostream>
#include "raymath.h"
using namespace std;
turnBlock::turnBlock()
{

	this->posX = GetMouseX();
	this->posY = GetMouseY();
	this->radius = 100;
	this->rotation = 180;
	float sq2 = sqrt(2.0f);
	this->A = { 0,0 };
	this->B = { 0,0 };
	this->C = { 0,0 };
	this->circleOrigin = { 0, 0};
	this->spread = 45.0f;
	this->points.resize(91);
	this->inFileType = 2;
	while (!WindowShouldClose()) {
		BeginDrawing();
		for (Blocks* klocek : klocki) {
			klocek->drawBlock();
		}
		
		this->posX = GetMouseX();
		this->posY = GetMouseY();
		
		if (IsKeyDown(KEY_A) && this->spread > 4) this->spread -= 3;
		if (IsKeyDown(KEY_D) && this->spread < 45) this->spread += 3;
		if (IsKeyDown(KEY_W) && this->radius > 10) this->radius += 5;
		if (IsKeyDown(KEY_S)) this->radius -= 5;
		if (IsKeyDown(KEY_Q)) this->rotation--;
		if (IsKeyDown(KEY_E)) this->rotation++;
		

		this->circleOrigin.x = this->posX - cos(DEG2RAD * this->rotation) * this->radius;
		this->circleOrigin.y = this->posY - sin(DEG2RAD * this->rotation) * this->radius;
		for (int i = 0; i < spread * 2 + 1; i++) {
			this->points[i].x = this->circleOrigin.x + cos(DEG2RAD * (this->rotation - this->spread + i)) * this->radius;
			this->points[i].y = this->circleOrigin.y + sin(DEG2RAD * (this->rotation - this->spread + i)) * this->radius;
		}
		for (int i = 0; i < this->spread * 2 ; i++) {
			DrawLineEx(this->points[i], this->points[i + 1], 16, WHITE);
			
		}
		
		
		
		

		this->A.x = this->circleOrigin.x + cos(DEG2RAD * (this->rotation + this->spread)) * this->radius;
		this->A.y = this->circleOrigin.y + sin(DEG2RAD * (this->rotation + this->spread)) * this->radius;

		this->B.x = this->circleOrigin.x + cos(DEG2RAD * (this->rotation)) * sq2 * this->radius;
		this->B.y = this->circleOrigin.y + sin(DEG2RAD * (this->rotation)) * sq2 * this->radius;

		this->C.x = this->circleOrigin.x + cos(DEG2RAD * (this->rotation - this->spread)) * this->radius;
		this->C.y = this->circleOrigin.y + sin(DEG2RAD * (this->rotation - this->spread)) * this->radius;
		
		
		ClearBackground({ backgroundColor });
		EndDrawing();
		if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
			
			break;
		}

		
	}
	
	


}

turnBlock::turnBlock(int x)
{

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
