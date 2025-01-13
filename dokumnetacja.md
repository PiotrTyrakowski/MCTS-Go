Przegląd Wysokopoziomowy
Silnik działa w następujących krokach:

Otrzymanie Pozycji: Użytkownik dostarcza aktualną pozycję na planszy.
Analiza przez MCTS: Silnik analizuje pozycję wykorzystując algorytm MCTS z akceleracją GPU.
Punktacja Obszarowa: Zastosowanie chińskiej punktacji do oceny wyniku gry.
Wybór Najlepszego Ruchu: Na podstawie analizy, silnik sugeruje optymalny ruch.


Zasady Punktacji
Chińska punktacja w Go polega na liczeniu liczby terytoriów oraz zajętych pól przez gracza. Każde terytorium jest sumowane wraz z policzeniem przechwyconych pionów przeciwnika.

Implementacja w Silniku
Silnik implementuje algorytm do identyfikacji i zliczania terytoriów na planszy:

Identifikacja Terytoriów: Algorytm przeszukuje planszę, identyfikuje obszary otoczone przez piony jednego gracza.
Liczenie Punktów: Sumowanie terytoriów i zajętych pól zgodnie z zasadami chińskiej punktacji.







Główne Fazy Algorytmu MCTS
MCTS składa się z czterech głównych faz, które są powtarzane iteracyjnie aż do osiągnięcia określonego limitu czasu lub liczby iteracji:

Selekcja (Selection)
Ekspansja (Expansion)
Symulacja (Simulation)
Backpropagacja (Backpropagation)
Poniżej znajduje się szczegółowe omówienie każdej z tych faz.

1. Selekcja (Selection)
W fazie selekcji algorytm przemieszcza się w dół drzewa MCTS, wybierając węzły na podstawie UCT. Celem jest znalezienie najbardziej obiecującego węzła do ekspansji.

Kroki:

Zaczynamy od korzenia drzewa (aktualnej pozycji na planszy).
Wybieramy dziecko węzła, które maksymalizuje wartość UCT.
Powtarzamy proces, aż dotrzemy do węzła, który nie został jeszcze w pełni rozwinięty (czyli ma możliwe ruchy do eksploracji).


2. Ekspansja (Expansion)
Jeśli węzeł wybrany do ekspansji nie jest liściem, czyli istnieją jeszcze możliwe ruchy, dodajemy jeden lub więcej nowych węzłów jako dzieci tego węzła.

Kroki:

Wybieramy jeden z nieodwiedzonych ruchów z aktualnego węzła.
Tworzymy nowy węzeł reprezentujący ten ruch.
Dodajemy nowy węzeł jako dziecko aktualnego węzła w drzewie MCTS.


3. Symulacja (Simulation)
Od nowo utworzonego węzła przeprowadzamy symulację  Wynik symulacji (wygrana, przegrana lub remis) służy do oceny jakości ruchu.

Kroki:

Rozpoczynamy od stanu gry reprezentowanego przez nowy węzeł.
Wykonujemy losowe ruchy aż do zakończenia gry.
Rejestrujemy wynik symulacji.


4. Backpropagacja (Backpropagation)
Wynik symulacji jest propagowany w górę drzewa MCTS, aktualizując statystyki odwiedzin i wygranych dla wszystkich węzłów na ścieżce od nowego węzła do korzenia.

Kroki:

Aktualizujemy licznik odwiedzin i licznik wygranych dla każdego węzła na ścieżce.
Jeśli symulacja zakończyła się zwycięstwem dla gracza, który reprezentuje dany węzeł, zwiększamy licznik wygranych.
Struktura Węzła w Drzewie MCTS
Każdy węzeł w drzewie MCTS reprezentuje określoną pozycję na planszy gry Go. Struktura węzła zawiera niezbędne informacje do efektywnego przeprowadzania algorytmu MCTS.




Struktura Węzła
Stan Gry (Game State)

Reprezentacja aktualnej pozycji na planszy.
Może zawierać informacje o wszystkich pionkach na planszy, kolejności ruchów, liczbie wykonanych ruchów itp.

Ruch (Move)

Ruch, który doprowadził do tego węzła od jego rodzica.
Reprezentowany jako para współrzędnych (np. E5).Kluczowe Elementy 

Rodzic (Parent)

Referencja do węzła nadrzędnego w drzewie.
Umożliwia nawigację w górę drzewa podczas backpropagacji.

Dzieci (Children)

Lista referencji do dzieci węzła.
Każde dziecko reprezentuje możliwy ruch z tego stanu gry.

Statystyki (Statistics)

Odwiedzenia (Visit Count): Liczba razy, kiedy węzeł został odwiedzony podczas fazy selekcji.
Wygrane (Win Count): Liczba wygranych symulacji przeprowadzonych z tego węzła.
Wartość (Value): Średnia wartość wygranych (np. wygrane / odwiedzenia).
Lista Nierozwiniętych Ruchów (Unexpanded Moves)

Lista możliwych ruchów, które jeszcze nie zostały dodane jako dzieci węzła.
Pomaga w fazie ekspansji do wyboru nowych ruchów do rozwinięcia.