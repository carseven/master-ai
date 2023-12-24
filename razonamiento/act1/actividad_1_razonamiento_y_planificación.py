# -*- coding: utf-8 -*-
"""Actividad 1 - Razonamiento y Planificación
# Actividad :  Resolución de problema mediante búsqueda heurística
"""
from __future__ import print_function


"""# 5.a.) El mapa, estado inicial y final de la figura (caso base)
"""

#!/usr/bin/env python
# coding: utf-8

# 2022 Modified by: Alejandro Cervantes
# Remember installing pyplot and flask if you want to use WebViewer

# NOTA: WebViewer sólo funcionará si ejecutáis en modo local

#Importa las funciones base
import math
from simpleai.search import SearchProblem, astar, breadth_first, depth_first
from simpleai.search.viewers import BaseViewer
#from simpleai.search.viewers import BaseViewer,ConsoleViewer,WebViewer


MAP = """
########
#    T #
# #### #
#   P# #
# ##   #
#      #
########
"""

MAP = [list(x) for x in MAP.split("\n") if x]

COSTS = {
    "up": 1.0,
    "down": 1.0,
    "right": 1.0,
    "left": 1.0,
}


class GameWalkPuzzle(SearchProblem):

    def __init__(self, board):
        self.board = board
        self.goal = (0, 0)
        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "t":
                    self.initial = (x, y)
                elif self.board[y][x].lower() == "p":
                    self.goal = (x, y)

        super(GameWalkPuzzle, self).__init__(initial_state=self.initial)

    def actions(self, state):
        actions = []
        for action in list(COSTS.keys()):
            newx, newy = self.result(state, action)
            if self.board[newy][newx] != "#":
                actions.append(action)
        return actions

    def result(self, state, action):
        x, y = state

        if action.count("up"):
            y -= 1
        if action.count("down"):
            y += 1
        if action.count("left"):
            x -= 1
        if action.count("right"):
            x += 1

        new_state = (x, y)
        return new_state

    def is_goal(self, state):
        return state == self.goal

    def cost(self, state, action, state2):
        return COSTS[action]

    # HEURÍSTICA EUCLÍDEA
    def euclidean_heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

    # HEURÍSTICA DE MANHATTAN
    def manhattan_heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        return abs(x - gx) + abs(y - gy)


def searchInfo (problem,result,use_viewer):
    def getTotalCost (problem,result):
        originState = problem.initial_state
        totalCost = 0
        for action,endingState in result.path():
            if action is not None:
                totalCost += problem.cost(originState,action,endingState)
                originState = endingState
        return totalCost

    result_path = result.path()
    res = f"Result path {result_path}\n"
    res += "Total length of solution: {0}\n".format(len(result_path))
    res += "Total cost of solution: {0}\n".format(getTotalCost(problem,result))

    if use_viewer:
        stats = [{'name': stat.replace('_', ' '), 'value': value}
                         for stat, value in list(use_viewer.stats.items())]

        for s in stats:
            res+= '{0}: {1}\n'.format(s['name'],s['value'])
    return res


def resultado_experimento(problem,MAP,result,used_viewer):
    path = [x[1] for x in result.path()]

    for y in range(len(MAP)):
        for x in range(len(MAP[y])):
            if (x, y) == problem.initial:
                print("T", end='')
            elif (x, y) == problem.goal:
                print("P", end='')
            elif (x, y) in path:
                print("·", end='')
            else:
                print(MAP[y][x], end='')
        print()

    info=searchInfo(problem,result,used_viewer)

    print(info)

def main():
    problem = GameWalkPuzzle(MAP)
    used_viewer=BaseViewer()

    #  Amplitud
    print("Búsqueda por Amplitud (BFS)")
    result = breadth_first(problem, graph_search=True,viewer=used_viewer)
    resultado_experimento(problem,MAP,result,used_viewer)

    #  Profundidad
    print("Búsqueda por Profundidad (DFS)")
    problem = GameWalkPuzzle(MAP)
    used_viewer=BaseViewer()
    result = depth_first(problem, graph_search=True,viewer=used_viewer)
    resultado_experimento(problem,MAP,result,used_viewer)

    #  Astar con heurística Euclídea
    print("Búsqueda por Astar con heurística Euclídea")
    problem = GameWalkPuzzle(MAP)
    problem.heuristic = problem.euclidean_heuristic
    used_viewer=BaseViewer()
    result = astar(problem, graph_search=True,viewer=used_viewer)
    resultado_experimento(problem,MAP,result,used_viewer)

    #  Astar con heurística de Manhattan
    print("Búsqueda por Astar con heurística de Manhattan")
    problem = GameWalkPuzzle(MAP)
    problem.heuristic = problem.manhattan_heuristic
    used_viewer = BaseViewer()
    result = astar(problem, graph_search=True, viewer=used_viewer)
    resultado_experimento(problem, MAP, result, used_viewer)

main()

"""# 5.b.) Una situación con un estado inicial modificado en el que el algoritmo de búsqueda en profundidad obtenga la solución óptima expandiendo menos nodos que el resto."""

MAP = """
########
#  T   #
# #### #
#   P# #
# ##   #
#      #
########
"""

MAP = [list(x) for x in MAP.split("\n") if x]

COSTS = {
    "up": 1.0,
    "down": 1.0,
    "right": 1.0,
    "left": 1.0,
}

main()

"""# 5.c.) Una tercera situación donse se use el estado inicial y final de 5.a, pero cambiando el coste del movimiento de la siguiente forma: los movimientos hacia abajo, izquierda y derecha tienen un coste de 1, mientras que los movimientos hacia arriba tienen un coste de 5."""

MAP = """
########
#    T #
# #### #
#   P# #
# ##   #
#      #
########
"""

MAP = [list(x) for x in MAP.split("\n") if x]

COSTS = {
    "up": 5.0,
    "down": 1.0,
    "right": 1.0,
    "left": 1.0,
}

main()

"""# 5.d) Una situación diseñada por el estudiante, con un mapa, un estado inicial y/o un estado final modificados, con coste unitario por acción (como en 5.a) , sobre el que pueda mostar y explicar las diferencias entre las dos heurísticas propuestas."""

MAP = """
################################
#                              #
#       ##                 P   #
#       ##   ########   ########
#       ##   #      ##         #
#       ##   #       #         #
#       ##   #      ###  #######
#       ##   #       #         #
#       ##   ##########        #
#       ## T                 ###
#       ##                   ###
#       ########################
################################
"""

MAP = [list(x) for x in MAP.split("\n") if x]

COSTS = {
    "up": 1.0,
    "down": 1.0,
    "right": 1.0,
    "left": 1.0,
}

main()