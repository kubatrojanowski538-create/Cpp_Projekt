#pragma once

#include "Util.h"


std::string GetText() {
	char text[100]{0};
	int size = 0;
	int character;
	WaitTime(0.5);
	while (GetCharPressed() > 0) {

	}
	while (!WindowShouldClose()) {
		BeginDrawing();

		character = GetCharPressed();
		if (character > 64 && character < 122 && size < 99) {
			text[size] = char(character);
			size++;
		}
		if (IsKeyPressed(KEY_BACKSPACE) && size > 0) {
			size--;
			text[size] = '\0';
			
		}
		if (IsKeyPressed(KEY_ENTER)) {
			EndDrawing();
			return std::string(text, size);
		}
		DrawText("nazwa pliku:", 10, 10, 20, WHITE);
		DrawText(text, 10, 40, 20, WHITE);
		ClearBackground(backgroundColor);
		EndDrawing();
	}
	return "";
}
Blocks* SetupBlock(int type) {
	
	int checkType = 1;
	float rot = 180;
	float len = 100;
	float spread = 45;
	int triggerType = 1;
	float posX{};
	float posY{};
	float startX{};
	float endX{};
	float startY{};
	float endY{};
	

	Vector2 start{};
	Vector2 end{};
	Vector2 circleOrigin{};
	Vector2 points[91]{};

	Color c = WHITE;

	if (type == 2) {
		len = 25;
	}



	while (!WindowShouldClose()) {
		BeginDrawing();
		ClearBackground(backgroundColor);
		for (Blocks* klocek : klocki) {
			klocek->drawBlock();
		}
		posX = float(GetMouseX());
		posY = float(GetMouseY());

		
		if (IsKeyDown(KEY_S) && len > 10) len -= 5;
		if (IsKeyDown(KEY_W)) len += 5;
		if (IsKeyDown(KEY_Q)) rot -= 3;
		if (IsKeyDown(KEY_E)) rot += 3;

		if(type == 1){
			startX = posX - cos(DEG2RAD * rot) * len / 2;
			endX = posX + cos(DEG2RAD * rot) * len / 2;
			startY = posY - sin(DEG2RAD * rot) * len / 2;
			endY = posY + sin(DEG2RAD * rot) * len / 2;

			start = { startX, startY };
			end = { endX, endY };

			DrawLineEx(start, end, 16, WHITE);

		
		}

		if (type == 2) {

			DrawCircle(posX, posY, len, WHITE);

		}

		if (type == 3) {
			if (IsKeyDown(KEY_A) && spread > 4) spread--;
			if (IsKeyDown(KEY_D) && spread < 45) spread++;
			circleOrigin.x = posX - cos(DEG2RAD * rot) * len;
			circleOrigin.y = posY - sin(DEG2RAD * rot) * len;
			for (int i = 0; i < spread * 2 + 1; i++) {
				points[i].x = circleOrigin.x + cos(DEG2RAD * (rot - spread + i)) * len;
				points[i].y = circleOrigin.y + sin(DEG2RAD * (rot - spread + i)) * len;
			}

			for (int i = 0; i < spread * 2; i++) {
				DrawLineEx(points[i], points[i + 1], 16, WHITE);

			}

		}

		if (type == 4) {
			if (IsKeyDown(KEY_A) && spread > 5) spread -= 5;
			if (IsKeyDown(KEY_D)) spread += 5;

			if (IsKeyPressed(KEY_ONE)) checkType = 1;
			if (IsKeyPressed(KEY_TWO)) checkType = 2;
			if (IsKeyPressed(KEY_THREE)) checkType = 3;

			if (checkType == 1) c = GREEN;
			if (checkType == 2) c = BLUE;
			if (checkType == 3) c = RED;
			DrawRectangle(posX - spread / 2, posY - len / 2, spread, len, c);
			DrawText(TextFormat("TYP: %d (1-Start, 2-Check, 3-Meta)", checkType), 10, 10, 20, WHITE);
		}



		if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
			break;
		}
		EndDrawing();
	}
	if (type == 1){
		BarrierLine* nowaLinia = new BarrierLine(len, rot, posX, posY, start, end, 0);
		return nowaLinia;
	}
	if (type == 2){
		pillarBlock* nowyPilar = new pillarBlock(posX, posY, len, 1);
		return nowyPilar;
	}
	if (type == 3){
		turnBlock* nowyZakret = new turnBlock(points, spread, 2);
		return nowyZakret;
	}
	if (type == 4) {
		TriggerBlock* nowyTrigger = new TriggerBlock(posX, posY, spread, len, checkType, 3);
		return nowyTrigger;
	}

	
	
}

Controls GetInputs()
{
	Controls inputs{};

	inputs.accelerate = IsKeyDown(KEY_W);
	inputs.brake = IsKeyDown(KEY_S);
	inputs.steerLeft = IsKeyDown(KEY_A);
	inputs.steerRight = IsKeyDown(KEY_D);

	return inputs;
}

std::ostream& operator<<(std::ostream& out, Vector2 Vec)
{
	return out << Vec.x << " " << Vec.y << " ";
}



std::istream& operator>>(std::istream& in, Vector2& Vec)
{
	return in >> Vec.x >> Vec.y;
}
