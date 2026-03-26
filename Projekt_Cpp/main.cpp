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
#include "Util.h"
using namespace std;





int main() {

    if (!DirectoryExists("tracks")) {
        MakeDirectory("tracks");
    }
    

    Image EKRAN;
    Color KOLORPIXELA;
    InitWindow(windowWidth, windowHeight, "cpp projekt v2");
    SetTargetFPS(fps);
    Car autko;

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(backgroundColor);
        for (Blocks* klocek : klocki) {
            klocek->drawBlock();
        }
        DrawText("1: Linia, 2: Zakret, 3: Filar, 4: START/META/CHECK, O: Zapisz ten tor do pliku, U: Wczytaj tor z pliku", 10, 10, 20, WHITE);
        DrawText("L: Jazda", 10, 40, 20, WHITE);
        EndDrawing();

        if (IsKeyPressed(KEY_ONE)) {
            BarrierLine* nowaLinia = new BarrierLine();
            klocki.push_back(nowaLinia);
        }
        if (IsKeyPressed(KEY_TWO)) {
            turnBlock* nowaLinia = new turnBlock();
            klocki.push_back(nowaLinia);
        }
        if (IsKeyPressed(KEY_THREE)) {
            pillarBlock* nowyPilar = new pillarBlock();
            klocki.push_back(nowyPilar);
        }
        if (IsKeyPressed(KEY_FOUR)) {
            TriggerBlock* nowyTrigger = new TriggerBlock();
            klocki.push_back(nowyTrigger);
        }
        if (IsKeyPressed(KEY_O)) {
            string NazwaPliku = GetText();
            string path = "tracks/";
            const string filePath = path + NazwaPliku + ".txt";
            ofstream plik;
            plik.open(filePath);
            for (Blocks* blok : klocki) {
                blok->writeBlock(plik);
            }

        }
        if (IsKeyPressed(KEY_U)) {
            string NazwaPliku = GetText();
            string path = "tracks/";
            const string filePath = path + NazwaPliku +".txt";
            fstream plik;
            plik.open(filePath);
            if (plik.fail()) {
                cout << "nie ma pliku";
            }
            klocki.clear();
            int temp;
            while (plik >> temp) {
                if (temp == 0) {
                    BarrierLine* nowy = new BarrierLine(1);
                    nowy->readBlock(plik);
                    klocki.push_back(nowy);
                }
                if (temp == 1) {

                    pillarBlock* nowy = new pillarBlock(1);
                    nowy->readBlock(plik);
                    klocki.push_back(nowy);
                }
                if (temp == 2) {

                    turnBlock* nowy = new turnBlock(1);
                    nowy->readBlock(plik);
                    klocki.push_back(nowy);
                }
                if (temp == 3) {

                    TriggerBlock* nowy = new TriggerBlock(1);
                    nowy->readBlock(plik);
                    klocki.push_back(nowy);
                }
                
            }


            
        }

        if (IsKeyPressed(KEY_L)) {
            break;
        }
    }

    drawScale = 5;

    for (Blocks* klocek : klocki) {
        klocek->scaleBlock();
        if (klocek->getBlockType() == 1) {
            autko.posX = klocek->posX;
            autko.posY = klocek->posY;
            respawnPoint = { klocek->posX, klocek->posY };
        }
    }

    camOffsetX = autko.posX - windowWidth / 2;
    camOffsetY = autko.posY - windowHeight / 2;

    isDrawing = false;

    while (!WindowShouldClose()) {

        if (timerRunning && !gameFinished) {
            gameTime += GetFrameTime();
        }
        
        BeginDrawing();

        for (Blocks* klocek : klocki) {
            klocek->drawBlock();
        }

        
        autko.drawCar();

        if (gameFinished) {
            DrawText(TextFormat("Czas: %.2f s", gameTime), windowWidth / 2 - 80, windowHeight / 2 + 20, 40, WHITE);

            if (IsKeyPressed(KEY_R)) {
                autko.resetCar(); 
            }
        }
        else {
            DrawText(TextFormat("Czas: %.2f", gameTime), 20, 20, 40, WHITE);
        }
        /////////////////////
        
        
        
        



        /////////////////////
        ClearBackground(backgroundColor);

        EndDrawing();
        
    }

    CloseWindow();
    return 0;
}