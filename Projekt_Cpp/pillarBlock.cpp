#include "pillarBlock.h"
#include "globals.h"

pillarBlock::pillarBlock()
{
	this->posX = GetMouseX();
	this->posY = GetMouseY();
	this->radius = 20;
	this->inFileType = 1;

	while (!WindowShouldClose()) {
		this->posX = GetMouseX();
		this->posY = GetMouseY();
		if (IsKeyDown(KEY_W)) this->radius++;
		if (IsKeyDown(KEY_S) && this->radius > 5) this->radius--;
		BeginDrawing();
		for (Blocks* klocek : klocki) {
			klocek->drawBlock();

		}
		DrawCircle(this->posX, this->posY, this->radius, WHITE);
		ClearBackground(backgroundColor);
		EndDrawing();
		if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
			break;
		}

	}

}

pillarBlock::pillarBlock(int x)
{

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
