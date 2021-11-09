#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:03:36 2021

@author: mesmaeeli
"""

import numpy as np
import matplotlib.pyplot as plt

class PATH_GAME:
    def __init__(self, grid_size, distribution, mode, start, end,**kwargs):
        """Possible distributions: normal, randint, logistic, poisson, chisquare"""
        kwargs = {'minimum':0,'maximum':10,'mean':0,'scale': 10,'lam':0,'df':2}
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
                ax2.text(j, i, '{0.1f}'.format(z), ha='center', va='center')
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        ax2.set_title('Game Grid')
        
        
        plt.show()
    
    def start_game(self):
        
        self.visited = []
        self.cost_list = []
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
        
        print('best path:',self.visited)
        print('cost:', sum(self.cost_list))
    
    def dijkstra(self):
        pass
    
    def find_neighbour(self,cell,direction):
        neighbours_list = []
        direction_list = {'UL':([-1, 0],[-1, 0]),
                          'UR':([0 , 1],[-1, 0]),
                          'LL':([-1, 0],[ 0, 1]),
                          'LR':([0 , 1],[ 0, 1])
                         }
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
        self.cost_list.append(self.play_ground[starting[0],starting[1]])
        adjacent = self.find_neighbour(starting, direct)
        
        if starting == ending:
            print('finished')
            return self.visited
        elif ending in adjacent:
            next_cord = ending
            next_value = self.play_ground[ending[0], ending[1]]
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

        return self.best_path(next_cord, ending,direct) 
 