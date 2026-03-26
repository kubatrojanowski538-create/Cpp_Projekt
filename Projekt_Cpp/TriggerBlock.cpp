#include "TriggerBlock.h"
#include "globals.h"
#include "raylib.h"
#include "raymath.h"

TriggerBlock::TriggerBlock() {
    this->posX = GetMouseX();
    this->posY = GetMouseY();
    this->width = 100;
    this->height = 100;
    this->type = 1; 
    this->inFileType = 3;
    while (!WindowShouldClose()) {
        this->posX = GetMouseX();
        this->posY = GetMouseY();

   
        if (IsKeyDown(KEY_W)) this->height += 2;
        if (IsKeyDown(KEY_S) && this->height > 10) this->height -= 2;
        if (IsKeyDown(KEY_D)) this->width += 2;
        if (IsKeyDown(KEY_A) && this->width > 10) this->width -= 2;

        if (IsKeyPressed(KEY_ONE)) this->type = 1;
        
        if (IsKeyPressed(KEY_TWO)) this->type = 2;

        if (IsKeyPressed(KEY_THREE)) this->type = 3;


        BeginDrawing();
        ClearBackground({ backgroundColor});

        for (Blocks* klocek : klocki) {
            klocek->drawBlock();
        }

        Color c = WHITE;
        if (this->type == 1) c = GREEN;
        if (this->type == 2) c = BLUE;
        if (this->type == 3) c = RED;

        DrawRectangle(this->posX - width / 2, this->posY - height / 2, width, height, c);
        DrawText(TextFormat("TYP: %d (1-Start, 2-Check, 3-Meta)", this->type), 10, 10, 20, WHITE);

        EndDrawing();

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            break;
        }
    }
}

TriggerBlock::TriggerBlock(int x)
{
}

void TriggerBlock::drawBlock() {
    Color c = WHITE;
    if (this->type == 1) c = { 0, 255, 0, 255 }; 
    if (this->type == 2) c = { 0, 0, 255, 255 }; 
    if (this->type == 3) c = { 255, 0, 0, 255 }; 

    DrawRectangle(
        (this->posX - width / 2) - camOffsetX,
        (this->posY - height / 2) - camOffsetY,
        width, height, c
    );
}

void TriggerBlock::scaleBlock() {
    this->posX *= drawScale;
    this->posY *= drawScale;
    this->width *= drawScale;
    this->height *= drawScale;
}

bool TriggerBlock::checkCollision(Vector2 point) {
    float left = this->posX - width / 2;
    float right = this->posX + width / 2;
    float top = this->posY - height / 2;
    float bottom = this->posY + height / 2;

    return (point.x >= left && point.x <= right && point.y >= top && point.y <= bottom);
}

int TriggerBlock::getBlockType() {
    return this->type;
}

void TriggerBlock::readBlock(std::istream& in)
{
    in >> this->type >> this->posX >> this->posY >> this->width >> this->height;
}

void TriggerBlock::writeBlock(std::ostream& out)
{
    out << this->inFileType << " " << this->type << " ";
    out << this->posX << " " << this->posY << " ";
    out << this->width << " " << this->height << std::endl;
}
