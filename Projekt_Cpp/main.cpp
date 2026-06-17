#include <iostream>
#include "raylib.h"
#include "math.h"
#include <ctime>
#include "globals.h"
#include "Car.h"
#include "BarrierLine.h"
#include "turnBlock.h"
#include "pillarBlock.h"
#include "TriggerBlock.h"
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include "Util.h"
#include "GameState.h"
using namespace std;


enum class GameScreen {
    TrackSelection,
    Driving,
    FinishMenu
};

struct TrackOption {
    string filePath;
    string displayName;
};

bool EndsWithTxt(const string& path) {
    if (path.size() < 4) return false;

    string extension = path.substr(path.size() - 4);
    for (char& character : extension) {
        character = static_cast<char>(tolower(static_cast<unsigned char>(character)));
    }

    return extension == ".txt";
}

string FileNameWithoutExtension(const string& path) {
    size_t slash = path.find_last_of("/\\");
    string fileName = slash == string::npos ? path : path.substr(slash + 1);
    size_t dot = fileName.find_last_of('.');

    if (dot != string::npos) {
        fileName = fileName.substr(0, dot);
    }

    return fileName;
}

vector<TrackOption> GetAvailableTracks() {
    vector<TrackOption> tracks;
    FilePathList files = LoadDirectoryFiles("tracks");

    for (unsigned int i = 0; i < files.count; i++) {
        string path = files.paths[i];

        if (EndsWithTxt(path)) {
            tracks.push_back({ path, FileNameWithoutExtension(path) });
        }
    }

    UnloadDirectoryFiles(files);

    sort(tracks.begin(), tracks.end(), [](const TrackOption& left, const TrackOption& right) {
        return left.displayName < right.displayName;
    });

    return tracks;
}

bool DrawButton(Rectangle bounds, const string& text, int fontSize) {
    Vector2 mousePosition = GetMousePosition();
    bool hovered = CheckCollisionPointRec(mousePosition, bounds);
    Color fillColor = hovered ? Color{ 80, 150, 105, 255 } : Color{ 35, 95, 65, 255 };

    DrawRectangleRec(bounds, fillColor);
    DrawRectangleLinesEx(bounds, 2, WHITE);
    DrawText(text.c_str(), static_cast<int>(bounds.x + 20), static_cast<int>(bounds.y + bounds.height / 2 - fontSize / 2), fontSize, WHITE);

    return hovered && IsMouseButtonReleased(MOUSE_BUTTON_LEFT);
}

void LoadBlock(fstream& file, int blockType) {
    if (blockType == 0) {
        BarrierLine* newBlock = new BarrierLine(1);
        newBlock->readBlock(file);
        klocki.push_back(newBlock);
    }

    if (blockType == 1) {
        pillarBlock* newBlock = new pillarBlock(1);
        newBlock->readBlock(file);
        klocki.push_back(newBlock);
    }

    if (blockType == 2) {
        turnBlock* newBlock = new turnBlock(1);
        newBlock->readBlock(file);
        klocki.push_back(newBlock);
    }

    if (blockType == 3) {
        TriggerBlock* newBlock = new TriggerBlock(1);
        newBlock->readBlock(file);
        klocki.push_back(newBlock);
    }
}

bool LoadTrack(const string& filePath) {
    fstream file;
    file.open(filePath);

    if (file.fail()) {
        return false;
    }

    klocki.clear();

    int blockType;
    while (file >> blockType) {
        LoadBlock(file, blockType);
    }

    return !klocki.empty();
}

void ResetRaceState(Car& car) {
    respawnPoint = { 0, 0 };
    gameTime = 0.0f;
    gameFinished = false;
    timerRunning = false;

    car.speed = 0;
    car.velX = 0;
    car.velY = 0;
    car.rotation = 0;
    car.respawnRot = 0;
}

void PrepareTrackForDriving(Car& car) {
    drawScale = 5;
    isDrawing = false;
    ResetRaceState(car);

    bool startFound = false;
    for (Blocks* block : klocki) {
        block->scaleBlock();

        if (block->getBlockType() == 1 && !startFound) {
            car.posX = block->posX;
            car.posY = block->posY;
            respawnPoint = { block->posX, block->posY };
            startFound = true;
        }
    }

    if (!startFound) {
        car.posX = 0;
        car.posY = 0;
    }

    camOffsetX = car.posX - windowWidth / 2;
    camOffsetY = car.posY - windowHeight / 2;
}

void ReturnToTrackSelection(Car& car) {
    klocki.clear();
    drawScale = 1;
    isDrawing = true;
    ResetRaceState(car);
    car.posX = 0;
    car.posY = 0;
    camOffsetX = 0;
    camOffsetY = 0;
}

void DrawTrackSelectionScreen(const vector<TrackOption>& tracks, string& loadError, GameScreen& screen, Car& car) {
    BeginDrawing();
    ClearBackground(backgroundColor);

    DrawText("Wybierz tor", windowWidth / 2 - 150, 80, 50, WHITE);
    DrawText("Kliknij jeden z dostepnych torow, aby rozpoczac gre.", windowWidth / 2 - 350, 150, 24, WHITE);

    if (!loadError.empty()) {
        DrawText(loadError.c_str(), windowWidth / 2 - 250, 190, 24, RED);
    }

    if (tracks.empty()) {
        DrawText("Brak plikow .txt w folderze tracks.", windowWidth / 2 - 250, 260, 28, WHITE);
    }

    const float buttonWidth = 520.0f;
    const float buttonHeight = 58.0f;
    const float startX = windowWidth / 2 - buttonWidth / 2;
    float y = 250.0f;

    for (const TrackOption& track : tracks) {
        Rectangle button = { startX, y, buttonWidth, buttonHeight };

        if (DrawButton(button, track.displayName, 28)) {
            if (LoadTrack(track.filePath)) {
                PrepareTrackForDriving(car);
                loadError.clear();
                screen = GameScreen::Driving;
            }
            else {
                loadError = "Nie udalo sie wczytac toru: " + track.displayName;
            }
        }

        y += buttonHeight + 14.0f;
    }

    EndDrawing();
}

void DrawDrivingScreen(Car& car, const string& gameStateFileName) {
    if (timerRunning && !gameFinished) {
        gameTime += GetFrameTime();
    }

    Controls inputs = GetInputs();
    car.UpdateGameState(inputs);

    if (!gameFinished) {
        AppendGameStateToFile(car.currentState, gameStateFileName);
    }

    car.UpdateRays();
    car.updateCar(inputs);

    BeginDrawing();
    ClearBackground(backgroundColor);

    for (Blocks* block : klocki) {
        block->drawBlock();
    }

    car.drawCar();
    DrawText(TextFormat("Czas: %.2f", gameTime), 20, 20, 40, WHITE);

    EndDrawing();
}

void DrawFinishMenu(Car& car, GameScreen& screen) {
    BeginDrawing();
    ClearBackground(backgroundColor);

    for (Blocks* block : klocki) {
        block->drawBlock();
    }

    car.drawCar();
    DrawRectangle(0, 0, windowWidth, windowHeight, Color{ 0, 0, 0, 170 });
    DrawText("Koniec gry", windowWidth / 2 - 160, windowHeight / 2 - 180, 52, WHITE);
    DrawText(TextFormat("Czas: %.2f s", gameTime), windowWidth / 2 - 130, windowHeight / 2 - 100, 38, WHITE);

    Rectangle retryButton = { windowWidth / 2 - 230.0f, windowHeight / 2 - 20.0f, 460.0f, 60.0f };
    Rectangle trackButton = { windowWidth / 2 - 230.0f, windowHeight / 2 + 60.0f, 460.0f, 60.0f };

    if (DrawButton(retryButton, "Jedz ten tor ponownie", 26)) {
        car.resetCar();
        gameTime = 0.0f;
        gameFinished = false;
        timerRunning = false;
        screen = GameScreen::Driving;
    }

    if (DrawButton(trackButton, "Wybierz nowy tor", 26)) {
        ReturnToTrackSelection(car);
        screen = GameScreen::TrackSelection;
    }

    EndDrawing();
}




int main() {

    if (!DirectoryExists("tracks")) {
        MakeDirectory("tracks");
    }
    
	string GameStateFileName = EnsureGameStateFileExists("GameStatesTable.csv");
    InitWindow(windowWidth, windowHeight, "cpp projekt v2");
    SetTargetFPS(fps);
    Car autko;
    GameScreen screen = GameScreen::TrackSelection;
    string loadError;

    while (!WindowShouldClose()) {
        if (screen == GameScreen::TrackSelection) {
            vector<TrackOption> tracks = GetAvailableTracks();
            DrawTrackSelectionScreen(tracks, loadError, screen, autko);
        }
        else if (screen == GameScreen::Driving) {
            DrawDrivingScreen(autko, GameStateFileName);

            if (gameFinished) {
                screen = GameScreen::FinishMenu;
            }
        }
        else if (screen == GameScreen::FinishMenu) {
            DrawFinishMenu(autko, screen);
        }
    }

    CloseWindow();
    return 0;
}