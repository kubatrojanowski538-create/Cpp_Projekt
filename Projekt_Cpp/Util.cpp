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

std::ostream& operator<<(std::ostream& out, Vector2 Vec)
{
	return out << Vec.x << " " << Vec.y << " ";
}



std::istream& operator>>(std::istream& in, Vector2& Vec)
{
	return in >> Vec.x >> Vec.y;
}
