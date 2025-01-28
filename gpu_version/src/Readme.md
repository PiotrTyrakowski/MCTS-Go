kompilacja programu:

nvcc -rdc=true main.cu position.cu mcts.cu neighbors.cu -o go_game_cu

plik constants.cuh zawiera zmienne ktore można zmienić:

N - wielkość planszy 

komi - wartość dodatnia dla białego niezależnie od pozycji.


5x5 - 5.5 komi

9x9 / 13x13 6.5 komi

19x19 7.5 komi 



podczas dzialania programu:

MCTS Iterations:  liczba iteracji algorytmu
MCTS Simulations: liczba gier puszczonych z jednego node'a.