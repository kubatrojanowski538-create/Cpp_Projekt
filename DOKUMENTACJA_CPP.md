# Dokumentacja projektu C++

## 1. Cel i zakres projektu

Projekt jest prostą grą samochodową napisaną w C++ z użyciem biblioteki
raylib. Część C++ odpowiada za uruchomienie okna gry, rysowanie toru,
sterowanie samochodem, wykrywanie kolizji, mierzenie czasu przejazdu oraz
zapisywanie stanu gry do pliku CSV. Repozytorium zawiera także tory zapisane w
plikach tekstowych oraz konfigurację projektu Visual Studio.

Ta dokumentacja opisuje stronę C++ projektu: strukturę plików, główne klasy,
format torów, przebieg pętli gry i sposób zapisu danych. Nie opisuje części
Pythonowej ani procesu trenowania modelu.

## 2. Wymagania i uruchamianie

Projekt C++ znajduje się w katalogu `Projekt_Cpp`. Plik rozwiązania Visual
Studio to `Projekt_Cpp3.sln`, a właściwy projekt to
`Projekt_Cpp/Projekt_Cpp.vcxproj`.

Najważniejsze wymagania:

- kompilator C++ obsługiwany przez Visual Studio,
- biblioteka raylib,
- katalog roboczy ustawiony tak, aby aplikacja widziała ścieżki względne:
  - `assets/car.png`,
  - `tracks/*.txt`,
  - `GameStatesTable.csv`.

Program używa ścieżek względnych, dlatego przy uruchamianiu z IDE albo z
konsoli należy zwrócić uwagę na katalog startowy procesu. Jeśli gra nie widzi
obrazka samochodu albo torów, najczęściej oznacza to, że katalog roboczy jest
ustawiony niezgodnie z układem projektu.

## 3. Struktura katalogu C++

Najważniejsze pliki:

- `main.cpp` - punkt wejścia programu, obsługa ekranów gry, wybór toru,
  uruchamianie jazdy i menu końca gry.
- `Car.h` / `Car.cpp` - logika samochodu: pozycja, prędkość, skręt, kolizje,
  promienie czujników oraz aktualny `GameState`.
- `Blocks.h` / `Blocks.cpp` - bazowa klasa elementów toru.
- `BarrierLine.*` - prosta linia bariery.
- `pillarBlock.*` - okrągła przeszkoda.
- `turnBlock.*` - zakręt zbudowany z wielu krótkich odcinków.
- `TriggerBlock.*` - prostokątny trigger: start, checkpoint albo meta.
- `Util.*` - funkcje pomocnicze, w tym pobieranie wejścia i obliczanie
  przecięć promieni z elementami toru.
- `GameState.*` - zapis nagłówka i kolejnych stanów gry do pliku CSV.
- `globals.*` - globalna konfiguracja okna, kamera, lista bloków i stan czasu.
- `tracks/` - pliki tekstowe z definicjami torów.
- `assets/` - zasoby graficzne, obecnie obraz samochodu.

## 4. Przebieg programu

Program startuje w funkcji `main()` w pliku `main.cpp`. Na początku sprawdzany
jest katalog `tracks`, tworzony jest plik `GameStatesTable.csv` z nagłówkiem
oraz inicjalizowane jest okno raylib. Następnie działa jedna główna pętla:

```cpp
while (!WindowShouldClose()) {
    // obsługa aktualnego ekranu gry
}
```

Aktualny ekran jest reprezentowany przez enum `GameScreen`:

- `TrackSelection` - ekran wyboru toru,
- `Driving` - właściwa jazda,
- `FinishMenu` - ekran po dojechaniu do mety.

Taki podział upraszcza powrót na początek pętli gry. Po zakończeniu przejazdu
program nie zamyka się, tylko przechodzi do menu końca. Z tego menu można
ponownie przejechać ten sam tor albo wrócić do listy torów i wybrać nowy.

## 5. Wybór toru

Tory są przechowywane w katalogu `tracks` jako pliki `.txt`. Na ekranie wyboru
program wywołuje `LoadDirectoryFiles("tracks")`, filtruje pliki z rozszerzeniem
`.txt`, sortuje je po nazwie i rysuje każdy tor jako przycisk.

Użytkownik wybiera tor kliknięciem myszy. Po kliknięciu wykonywane są kroki:

1. Otwarcie pliku toru.
2. Wyczyszczenie aktualnej listy bloków `klocki`.
3. Odczyt kolejnych elementów toru.
4. Skalowanie toru do trybu jazdy.
5. Ustawienie samochodu na triggerze startu.
6. Przejście do ekranu `Driving`.

Wczytywanie toru jest realizowane przez funkcję `LoadTrack()`. Każdy wpis w
pliku zaczyna się od liczby określającej typ zapisanego obiektu. Funkcja
`LoadBlock()` tworzy odpowiednią klasę C++ i wywołuje jej `readBlock()`.

## 6. Format torów

Format plików w `tracks` jest tekstowy. Każdy obiekt zapisuje najpierw swój typ
plikowy (`inFileType`), a potem własne dane geometryczne.

Typy zapisu używane przy wczytywaniu:

- `0` - `BarrierLine`, czyli odcinek bariery,
- `1` - `pillarBlock`, czyli okrągła przeszkoda,
- `2` - `turnBlock`, czyli zakręt opisany tablicą punktów,
- `3` - `TriggerBlock`, czyli prostokąt specjalny.

`TriggerBlock` ma dodatkowe pole `type`, które określa znaczenie w grze:

- `1` - start,
- `2` - checkpoint,
- `3` - meta.

Warto odróżnić typ zapisu w pliku od typu kolizji zwracanego przez
`getBlockType()`. Bariery, filary i zakręty są dla samochodu przeszkodami,
natomiast trigger może być startem, checkpointem albo metą.

## 7. Bloki toru

Wszystkie elementy toru dziedziczą po klasie `Blocks`, która definiuje wspólny
interfejs:

- `drawBlock()` - rysowanie elementu,
- `scaleBlock()` - przeskalowanie po wczytaniu toru,
- `checkCollision(Vector2 point)` - sprawdzenie kolizji punktu z blokiem,
- `readBlock()` i `writeBlock()` - odczyt i zapis do pliku,
- `getBlockType()` - typ wykorzystywany w logice kolizji.

`BarrierLine` reprezentuje ścianę jako odcinek. Kolizja jest liczona przez
rzutowanie punktu na odcinek i sprawdzenie odległości od tego odcinka.

`pillarBlock` reprezentuje okrąg. Kolizja jest spełniona, kiedy odległość
punktu od środka jest mniejsza niż promień.

`turnBlock` zapisuje zakręt jako serię punktów. Rysowanie i kolizje są liczone
dla kolejnych odcinków między punktami.

`TriggerBlock` jest prostokątem. W zależności od typu jest rysowany innym
kolorem i może oznaczać start, checkpoint albo metę.

## 8. Samochód i sterowanie

Klasa `Car` przechowuje pozycję, prędkość, rotację, teksturę samochodu,
prostokąt rysowania oraz promienie czujników. Sterowanie jest pobierane przez
funkcję `GetInputs()` z pliku `Util.cpp`:

- `W` - przyspieszanie,
- `S` - hamowanie / cofanie,
- `A` - skręt w lewo,
- `D` - skręt w prawo.

Metoda `updateCar()` wykonuje trzy najważniejsze operacje:

1. `updateSpeedRot()` - aktualizuje prędkość i rotację na podstawie wejścia,
2. `updatePosition()` - przesuwa samochód i aktualizuje kamerę,
3. `checkCollision()` - sprawdza kontakt z elementami toru.

Kolizja samochodu jest liczona przez kilka punktów umieszczonych wokół bryły
samochodu. Każdy punkt jest porównywany z każdym blokiem toru. Jeśli samochód
dotknie przeszkody, wraca na aktualny punkt respawnu. Jeśli dotknie checkpointu,
punkt respawnu jest aktualizowany. Jeśli dotknie mety, gra ustawia
`gameFinished = true` i zatrzymuje licznik.

## 9. Kamera, skala i rysowanie

Po wczytaniu toru ustawiana jest globalna skala `drawScale = 5`. Bloki są
skalowane metodą `scaleBlock()`, a następnie rysowane względem przesunięcia
kamery `camOffsetX` i `camOffsetY`.

Kamera jest związana z samochodem. Pozycja kamery jest aktualizowana tak, aby
samochód znajdował się w okolicy środka okna:

```cpp
camOffsetX = car.posX - windowWidth / 2;
camOffsetY = car.posY - windowHeight / 2;
```

Rysowanie odbywa się w funkcjach ekranów. Dla trybu jazdy najpierw czyszczone
jest tło, potem rysowane są bloki toru, samochód i licznik czasu.

## 10. Promienie czujników i zapis stanu gry

Samochód generuje 20 promieni czujników. Promienie są liczone względem rotacji
samochodu. Część promieni patrzy przed samochód, a część szerzej wokół niego.

Metoda `UpdateGameState()` zapisuje do struktury:

- aktualną prędkość,
- dystans trafienia dla każdego promienia,
- typ obiektu trafionego przez promień,
- wejścia gracza w danej klatce.

Funkcje z `GameState.cpp` zapisują dane do `GameStatesTable.csv`. Jeśli plik
nie istnieje albo jest pusty, `EnsureGameStateFileExists()` tworzy nagłówek.
Podczas jazdy, dopóki gra nie jest zakończona, każda klatka dopisuje kolejny
wiersz przez `AppendGameStateToFile()`.

Z perspektywy C++ ten plik jest tylko eksportem danych z gry. Kod C++ nie
trenuje modelu, tylko przygotowuje dane: stan samochodu, odczyty promieni i
wejścia sterujące.

## 11. Menu końca gry

Po dotknięciu mety samochód ustawia `gameFinished = true`. Główna pętla
przechodzi wtedy z ekranu `Driving` do `FinishMenu`.

Menu końca pokazuje:

- napis końca gry,
- końcowy czas przejazdu,
- przycisk ponownego przejazdu tego samego toru,
- przycisk powrotu do wyboru nowego toru.

Ponowny przejazd resetuje czas, prędkość, rotację i pozycję samochodu do
startu bieżącego toru. Powrót do wyboru toru czyści listę bloków, resetuje
kamerę i przywraca ekran listy dostępnych torów.

## 12. Najważniejsze zmienne globalne

Zmienne globalne są zdefiniowane w `globals.cpp` i zadeklarowane w
`globals.h`. Najważniejsze z nich:

- `windowWidth`, `windowHeight` - rozmiar okna,
- `fps` - docelowa liczba klatek na sekundę,
- `klocki` - lista wszystkich bloków aktualnego toru,
- `drawScale` - skala rysowania i geometrii po wczytaniu toru,
- `isDrawing` - informacja, czy gra jest w trybie edycji/rysowania,
- `backgroundColor` - kolor tła,
- `respawnPoint` - punkt powrotu samochodu,
- `gameTime` - czas aktualnego przejazdu,
- `gameFinished` - informacja, czy samochód dojechał do mety,
- `timerRunning` - informacja, czy licznik czasu działa,
- `maxRayDistance` - maksymalny zasięg promieni czujników.

## 13. Kierunki dalszego rozwoju części C++

Kod C++ można dalej rozwijać bez zmieniania części Pythonowej. Najbardziej
naturalne kierunki to:

- uporządkowanie zarządzania pamięcią bloków toru,
- przeniesienie ekranów gry z `main.cpp` do osobnych klas lub modułów,
- dodanie przewijania listy torów, jeśli liczba plików w `tracks` wzrośnie,
- lepsza obsługa błędów przy uszkodzonych plikach torów,
- ujednolicenie nazewnictwa klas i metod,
- dodanie automatycznych testów dla parsera torów i funkcji geometrycznych.

Obecnie najważniejsza odpowiedzialność części C++ to utrzymanie stabilnej pętli
gry, poprawne wczytywanie torów, obsługa kolizji i generowanie danych stanu gry.
