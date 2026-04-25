
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

Vector2 NormalizeVector2(Vector2 vec)
{
	float len = sqrt(pow(vec.x, 2 ) + pow(vec.y, 2));

	return {vec.x / len, vec.y / len};
}




float cross(Vector2 a, Vector2 b) {
	return a.x * b.y - a.y * b.x;
}



float RayDistance2D(Vector2 P, Vector2 D, Vector2 A, Vector2 B) {
	Vector2 r = D;
	Vector2 s = { B.x - A.x, B.y - A.y };

	float rxs = cross(r, s);
	if (fabs(rxs) < 0.0001f) return maxRayDistance; 

	Vector2 AP = { A.x - P.x, A.y - P.y };

	float t = cross(AP, s) / rxs;
	float u = cross(AP, r) / rxs;


	
	if (t >= 0 && u >= 0 && u <= 1) {
		
		return t;
		;
	}

	return maxRayDistance;
}
float RayDistance2DPillar(Vector2 P, Vector2 D, pillarBlock* pillar)
{
	Vector2 C = { pillar->posX, pillar->posY };
	float R = pillar->radius;

	Vector2 PC = { P.x - C.x, P.y - C.y };

	float a = D.x * D.x + D.y * D.y;
	if (fabs(a) < 0.0001f) return maxRayDistance;

	float b = 2.0f * (PC.x * D.x + PC.y * D.y);
	float c = PC.x * PC.x + PC.y * PC.y - R * R;

	float delta = b * b - 4.0f * a * c;
	if (delta < 0.0f) return maxRayDistance;

	float sqrtDelta = sqrt(delta);

	float t1 = (-b - sqrtDelta) / (2.0f * a);
	float t2 = (-b + sqrtDelta) / (2.0f * a);

	if (t1 >= 0.0f) return t1;
	if (t2 >= 0.0f) return t2;

	return maxRayDistance;
}
float RayDistance2DTrigger(Vector2 P, Vector2 D, TriggerBlock* trigger)
{
	float left = trigger->posX - trigger->width / 2.0f;
	float right = trigger->posX + trigger->width / 2.0f;
	float top = trigger->posY - trigger->height / 2.0f;
	float bottom = trigger->posY + trigger->height / 2.0f;

	float tMin = -maxRayDistance;
	float tMax = maxRayDistance;

	if (fabs(D.x) < 0.0001f) {
		if (P.x < left || P.x > right) return maxRayDistance;
	}
	else {
		float tx1 = (left - P.x) / D.x;
		float tx2 = (right - P.x) / D.x;

		if (tx1 > tx2) {
			float tmp = tx1;
			tx1 = tx2;
			tx2 = tmp;
		}

		if (tx1 > tMin) tMin = tx1;
		if (tx2 < tMax) tMax = tx2;
	}

	if (fabs(D.y) < 0.0001f) {
		if (P.y < top || P.y > bottom) return maxRayDistance;
	}
	else {
		float ty1 = (top - P.y) / D.y;
		float ty2 = (bottom - P.y) / D.y;

		if (ty1 > ty2) {
			float tmp = ty1;
			ty1 = ty2;
			ty2 = tmp;
		}

		if (ty1 > tMin) tMin = ty1;
		if (ty2 < tMax) tMax = ty2;
	}

	if (tMax < 0.0f) return maxRayDistance;
	if (tMin > tMax) return maxRayDistance;

	if (tMin < 0.0f) return 0.0f;

	return tMin;
}
float RayDistance2DTurn(Vector2 P, Vector2 D, turnBlock* turn)
{
	float closest = maxRayDistance;

	for (int i = 0; i < turn->spread * 2; i++) {
		float dist = RayDistance2D(P, D, turn->points[i], turn->points[i + 1]);

		if (dist < closest) {
			closest = dist;
		}
	}

	return closest;
}

std::ostream& operator<<(std::ostream& out, Vector2 Vec)
{
	return out << Vec.x << " " << Vec.y << " ";
}



std::istream& operator>>(std::istream& in, Vector2& Vec)
{
	return in >> Vec.x >> Vec.y;
}
