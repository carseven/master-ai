#!/usr/bin/env python
# coding: utf-8

# 2021 Modified by: Alejandro Cervantes
# Remember installing pyplot and flask if you want to use WebViewer

from __future__ import print_function
from ast import arg

import math
import sys
from types import NoneType
from simpleai.search import SearchProblem, astar, breadth_first, depth_first
from simpleai.search.viewers import BaseViewer,ConsoleViewer,WebViewer


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

MAP_5_b = """
########
#  T   #
# #### #
#   P# #     
# ##   #
#      #
########
"""
MAP_5_b = [list(x) for x in MAP_5_b.split("\n") if x]

MAP_5_d = """
########
#     P#
# ###  #
#   #  #     
# ##   #
#T     #
########
"""
MAP_5_d = [list(x) for x in MAP_5_d.split("\n") if x]


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

    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)
    
    def heuristic_Manhattan(self, state):
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

    print ("Result")
    print (result.path())
    
    res = "Total length of solution: {0}\n".format(len(result.path()))
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

def ejecutar_BFS(map):
    problem = GameWalkPuzzle(map)
    used_viewer=BaseViewer() 
    result = breadth_first(problem, graph_search=True,viewer=used_viewer)
    resultado_experimento(problem,map,result,used_viewer)

def ejecutar_DFS(map):
    problem = GameWalkPuzzle(map)
    used_viewer=BaseViewer() 
    result = depth_first(problem, graph_search=True,viewer=used_viewer)
    resultado_experimento(problem,map,result,used_viewer)

def ejecutar_A(map, useManhattan = False):
    problem = GameWalkPuzzle(map)
    used_viewer=BaseViewer() 
    
    if (useManhattan is True):
        problem.heuristic = problem.heuristic_Manhattan

    result = astar(problem, graph_search=True,viewer=used_viewer)
    resultado_experimento(problem,map,result,used_viewer) 

def ejercicio_5_a():
    print(" \nEjecutando ejercicio A: \n") 

    print ("\n BFS \n")
    ejecutar_BFS(MAP)    
        
    print ("DFS \n")
    ejecutar_DFS(MAP)
 
    print ("A* \n")
    ejecutar_A(MAP)

    print ("A* con heuristica Manhattan \n")
    ejecutar_A(MAP,True)


    print ("\n --------------")

def ejercicio_5_b():
    print(" \nEjecutando ejercicio B: \n")

    print ("\n BFS con mapa modificado\n")
    ejecutar_BFS(MAP_5_b)   
        
    print ("DFS con mapa modificado\n")
    ejecutar_DFS(MAP_5_b)

    print ("A* con mapa modificado\n")
    ejecutar_A(MAP_5_b)

    print ("\n A* con heuristica Manhattan\n")
    # Se utiliza la heuristica de mahnattan, tal y como solicitan en el enunciado   
    ejecutar_A(MAP_5_b, True)  
    
def ejercicio_5_c():
    print(" \nEjecutando ejercicio C: \n")

    # Se actualiza el coste, tal y como se solicita en el enunciado
    COSTS["up"] = 5

    print ("\n BFS con coste up a 5\n")
    ejecutar_BFS(MAP)   
        
    print ("DFS con coste up a 5\n")
    ejecutar_DFS(MAP)

    print ("\n A* con coste up a 5\n")
    ejecutar_A(MAP)

    print ("A* con heuristica Manhattan \n")
    ejecutar_A(MAP,True)

    print ("\n --------------")

def ejercicio_5_d():
    print(" \nEjecutando ejercicio D: \n")

    print ("\n A* \n")
    ejecutar_A(MAP_5_d) 
     
    print ("\n A* con heuristica Manhattan\n")
    # Se utiliza la heuristica de mahnattan, tal y como solicitan en el enunciado   
    ejecutar_A(MAP_5_d, True)  
    
    print ("\n --------------")


def main():
    args = sys.argv[1:] 

    if (len(args) == 0):      
        args.append('all')

    match args[0]:
        case 'all':
            print(" \n => Ejecutando todos los ejercicios\n")
            ejercicio_5_a()
            ejercicio_5_b()
            ejercicio_5_c()
            ejercicio_5_d()
        case 'a':          
            ejercicio_5_a()
        case 'b':
            ejercicio_5_b()
        case 'c':
            ejercicio_5_c()
        case 'd':
            ejercicio_5_d() 
 

if __name__ == "__main__":
    main()
