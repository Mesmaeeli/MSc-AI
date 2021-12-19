#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:03:36 2021

@author: mesmaeeli
"""


import numpy as np
import matplotlib.pyplot as plt

class PATH_GAME:
    def __init__(self, grid_size, distribution, mode, start, end, game_type, **kwargs):
        """Possible distributions: normal, randint, logistic, poisson, chisquare"""
        kwargs = {'minimum':0,'maximum':10,'mean':10,'scale': 10,'lam':10,'df':2}
        self.distr = getattr(np.random,distribution)
        
        if distribution == 'normal' or distribution == 'logistic':
            self.Game = self.distr(loc = kwargs['mean'], scale = kwargs['scale'],size = grid_size)
        elif distribution == 'randint':
            self.Game = self.distr(kwargs['minimum'],kwargs['maximum'],size = grid_size)
        elif distribution == 'poisson':
            self.Game = self.distr(kwargs['lam'],size = grid_size)
        else:
            self.Game = self.distr(kwargs['df'],size = grid_size)
        
        
        self.start_node = start
        self.end_node = end
        self.mode = mode
        self.visited = []
        self.cost_list = []
        self.play_ground = np.arange(self.Game.size).reshape(self.Game.shape[0],-1).astype('object')
        self.play_ground[:] = np.Inf
        
        if game_type == 'dijkstra':
            self.dijkstra(start, end)
        elif game_type == 'naive':
            self.start_game()
        else:
            print('Invalid game approach')
        
        self.visualize()

    def cost_func(self,current,neighbour):
        if self.mode == 1:
            return self.play_ground[neighbour[0],neighbour[1]]      
        elif self.mode == 2:
            return abs(self.play_ground[current[0],current[1]] - self.play_ground[neighbour[0],neighbour[1]])

    
    def visualize(self):
        x = []
        y = []
        for i,j in self.visited:
            x.append(i)
            y.append(j)
        z = 0
        dx = dy = 0.5
        dz = self.cost_list

        
        fig = plt.figure()
        fig.suptitle('Possible Shortest Path: {}'.format(sum(self.cost_list)))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.bar3d(x, y, z, dx, dy, dz, shade=True)
        ax1.set_title('Path Cost')

        ax2 = fig.add_subplot(122)
        data = np.zeros(self.Game.shape)
        for cell in self.visited:
            data[cell[0],cell[1]] = 1
        
        ax2.matshow(data, cmap = 'summer')
        
        if self.Game.dtype == 'int':
            for (i, j), z in np.ndenumerate(self.Game):
                ax2.text(j, i, '{}'.format(z), ha='center', va='center')
        else:
            for (i, j), z in np.ndenumerate(self.Game):
                ax2.text(j, i, format(z,'.0f'), ha='center', va='center')
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        ax2.set_title('Game Grid')
        
        
        plt.show()
    
    def start_game(self):
        
        self.visited = []
        self.cost_list = [self.Game[self.start_node[0],self.start_node[1]]]
        a1 = min(self.start_node[0],self.end_node[0])
        a2 = max(self.start_node[0],self.end_node[0])
        b1 = min(self.start_node[1],self.end_node[1])
        b2 = max(self.start_node[1],self.end_node[1])
        for i in range(a1, a2 + 1):
            for j in range(b1, b2 + 1):
                self.play_ground[i,j] = self.Game[i,j]
        
        if self.start_node[0] <= self.end_node[0] and self.start_node[1] <= self.end_node[1]:
            direct = 'LR'
        elif self.start_node[0] > self.end_node[0] and self.start_node[0] < self.end_node[0]:
            direct = 'LL'
        elif self.start_node[0] <= self.end_node[0] and self.start_node[1] >= self.end_node[1]:
            direct = 'UR'
        else:
            direct = 'UL'
        
        self.best_path(self.start_node, self.end_node, direct)
        
        # print('best path:',self.visited)
        # print('cost:', sum(self.cost_list))
        return sum(self.cost_list)
    
    
    def find_neighbour(self,cell,direction = None):
        neighbours_list = []
        

        direction_list = {'UL':([-1, 0],[-1, 0]),
                          'UR':([0 , 1],[-1, 0]),
                          'LL':([-1, 0],[ 0, 1]),
                          'LR':([0 , 1],[ 0, 1])
                         }
        
        if direction == None:
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    if (i * j == 1 or i * j == -1) or (i == 0 and j == 0):
                        continue
                    try:
                        neighbour = [cell[0]+i,cell[1]+j]
                        assert self.Game.shape[0] > cell[0]+i >= 0 and self.Game.shape[1] > cell[1]+j >= 0
                        neighbours_list.append(neighbour)                
                    except:
                        continue
            return neighbours_list
            
        else:
            for i in direction_list[direction][0]:
                for j in direction_list[direction][1]:
                    if (i * j == 1 or i * j == -1) or (i == 0 and j == 0):
                        continue
                    try:
                        neighbour = [cell[0]+i,cell[1]+j]
                        assert self.Game.shape[0] > cell[0]+i >= 0 and self.Game.shape[1] > cell[1]+j >= 0
                        neighbours_list.append(neighbour)                
                    except:
                        continue
            return neighbours_list
    
    def best_path(self,starting,ending,direction):

        direct = direction
        self.visited.append(starting)
        # self.cost_list.append(self.play_ground[starting[0],starting[1]])
        adjacent = self.find_neighbour(starting, direct)
        
        if starting == ending:
            # print('finished')
            return self.visited
        elif ending in adjacent:
            next_cord = ending
            # next_value = self.play_ground[ending[0], ending[1]]
            next_value = self.cost_func(starting, ending)
        else:
            
            next_cord = []
            next_value = np.Inf
            for cell in adjacent:
                # next_cord , next_value = method1(cell,next_value)
                # if self.play_ground[cell[0],cell[1]] < next_value:
                #     next_cord = cell
                #     next_value = play_ground[cell[0],cell[1]] 
                if self.cost_func(starting, cell) < next_value:
                    next_cord = cell
                    next_value = self.cost_func(starting, cell)
        self.cost_list.append(next_value)
        return self.best_path(next_cord, ending,direct) 
    
    def dijkstra(self, starting, ending):
        
        self.source = starting[0] * self.Game.shape[0]  + starting[1]
        self.target = ending[0] * self.Game.shape[0]  + ending[1]
        self.parents = {}
        self.graph = {}
        self.cost = {}
        self.play_ground = self.Game
        for rows in range(self.Game.shape[0]):
            for cols in range(self.Game.shape[1]):
                code = rows * self.Game.shape[0] + cols
                self.graph[code] = {}
                if starting == [rows,cols]:
                    self.cost[code] = 0
                else:
                    self.cost[code] = np.Inf
                for cell in self.find_neighbour([rows,cols]):
                    cell_code = cell[0] * self.Game.shape[0] + cell[1]
                    self.graph[code][cell_code] = self.cost_func([rows,cols], [cell[0],cell[1]])
    
        nextNode = self.source
        
        while nextNode != self.target:
            
            for neighbor in self.graph[nextNode]:               
                if self.graph[nextNode][neighbor] + self.cost[nextNode] < self.cost[neighbor]:                   
                    self.cost[neighbor] = self.graph[nextNode][neighbor] + self.cost[nextNode]                   
                    self.parents[neighbor] = nextNode                   
                del self.graph[neighbor][nextNode]              
            del self.cost[nextNode]           
            nextNode = min(self.cost, key=self.cost.get)
            
        node = self.target      
        backpath = [self.target]      
        path = []
        
        while node != self.source:          
            backpath.append(self.parents[node])       
            node = self.parents[node]
            
        for i in range(len(backpath)):     
            path.append(backpath[-i - 1])
        final = []
        final_cost = 0
        self.cost_list = []
        for i in path:
            
            final.append([i // self.Game.shape[0], i % self.Game.shape[0]])
            self.cost_list.append(self.Game[i // self.Game.shape[0],i % self.Game.shape[0]])
            final_cost += self.Game[i // self.Game.shape[0],i % self.Game.shape[0]]

        if self.mode == 2:
            a = np.array(self.cost_list[:-1])
            b = np.array(self.cost_list[1:])
            
            final_cost = np.sum(np.abs(a-b))
            self.cost_list = np.abs(a-b)
 
            self.cost_list = np.append(self.cost_list,0)
        self.visited = final    
  

        return final
        
           
    def to_be_visited(self):
      next_vertex = -100
      # Choosing the vertex with the minimum distance
      for index in range(self.number_of_vertices):
        if self.visited_and_distance[index][0] == 0 \
          and (next_vertex < 0 or self.visited_and_distance[index][1] <= \
          self.visited_and_distance[next_vertex][1]):
            next_vertex = index
      return next_vertex
        
    
 
def Game_analysis():
      
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    all_costs = []
    x = []
    for i in range(2,16):
        G = PATH_GAME((i,i),'randint',2,[0,0],[i-1,i-1],minimum = 2,maximum = 20)
        all_costs.append(G.start_game())
        x.append(i)
        
    ax1.bar(x, all_costs)
    
    ax2 = fig.add_subplot(122)
    dists = ['normal', 'randint', 'logistic', 'poisson', 'chisquare']
    all_costs = []
    for dist in dists:
        G = PATH_GAME((10,10),dist,1,[0,0],[9,9])
        all_costs.append(G.start_game())
        
    ax2.bar([0,1,2,3,4],all_costs)
        
    
    
    plt.show()
np.random.seed(2)
G = PATH_GAME((10,10),'randint',2,[0,0],[9,9], 'dijkstra')


            
            
