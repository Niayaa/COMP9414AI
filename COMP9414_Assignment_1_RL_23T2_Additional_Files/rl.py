# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:57:23 2023

@author: Francisco

Reinforcement learning

"""

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

np.random.seed(9999)
random_numbers=np.random.random(100000)
np.savetxt("random_numbers.txt", random_numbers)


class World(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.R = np.zeros(self.x*self.y)
        self.agentPos = 0

    def idx2xy(self,idx):
        x = int(idx / self.y)
        y = idx % self.y
        return x, y

    def xy2idx(self,x,y):
        return x*self.y + y

    def resetAgent(self, pos):
        self.agentPos = int(pos)

    def setReward(self, x, y, r):
        goalState = self.xy2idx(x, y)
        self.R[goalState] = r

    def getState(self):
        return self.agentPos

    def getReward(self):
        return self.R[self.agentPos]

    def getNumOfStates(self):
        return self.x*self.y

    def getNumOfActions(self):
        return 4

    def move(self,id):
        x_, y_ = self.idx2xy(self.agentPos)
        tmpX = x_
        tmpY = y_
        if id == 0: # move DOWN
            tmpX += 1
        elif id == 1: # move UP
            tmpX -= 1
        elif id == 2: # move RIGHT
            tmpY += 1
        elif id == 3: # move LEFT
            tmpY -= 1
        else:
            print("ERROR: Unknown action")

        if self.validMove(tmpX, tmpY):
            self.agentPos = self.xy2idx(tmpX,tmpY)

    def validMove(self,x,y):
        valid = True
        if x < 0 or x >= self.x:
            valid = False
        if y < 0 or y >= self.y:
            valid = False
        return valid

class Agent(object):
    def __init__(self, world):
        self.world = world
        self.numOfActions = self.world.getNumOfActions()
        self.numOfStates = self.world.getNumOfStates()

        self.alpha = 0.7
        self.gamma = 0.4
        self.epsilon = 0.25
        self.temp = 0.1
        #self.Q = np.random.uniform(0.0,0.01,(self.numOfStates,self.numOfActions))
        self.Q = np.loadtxt("initial_Q_values.txt")#initializing the Q-matrix


        self.random_numbers = np.loadtxt("random_numbers.txt")  # Load random numbers
        self.random_counter = 0


    # epsilon-greedy action selection
    def actionSelection(self, state):
        random_number = self.random_numbers[self.random_counter]
        self.random_counter += 1
        random_number2 = self.random_numbers[self.random_counter]
        #self.random_counter += 1
        #print("before "+str(self.random_counter))
        action=0
        if random_number <= self.epsilon:
            # Exploration: Choose a random action
            if random_number2 <= 0.25:
                action = 0  # down
            elif random_number2 <= 0.5:
                action = 1  # up
            elif random_number2 <= 0.75:
                action = 2  # right
            else:
                action = 3  # left
            self.random_counter += 1
        else:
            # Exploitation: Choose the action with the highest Q-value
            action = np.argmax(self.Q[state, :])

        return action

    def update(self,s,s1,r,a,a1):
        predict = self.Q[s,a]
        tar = r + self.gamma * self.Q[s1,a1]
        self.Q[s,a]=self.Q[s,a]+ self.alpha * (tar-predict)

    def softmaxSelection(self,state):
        prob = []
        #weight = np.array([self.Q[(state,x)]/self.temp for x in range(self.numOfActions)])
        #prob_a = np.exp(weight)/np.sum(np.exp(weight))
        z = sum([np.exp(self.Q[(state,x)]/self.temp)for x in range(self.numOfActions)])
        prob_a = [np.exp(self.Q[(state,x)]/self.temp)/z for x in range(self.numOfActions)]
        #cum_prob = 0.0   #cumweight for exchange weight by equation cumweight = weightofprevious+itsownweight
        select = self.random_numbers[self.random_counter] #the range from 0 to sum of probability which is 1
        # print(type(prob_a))
        for i in range(len(prob_a)):
            prob.append(np.sum(prob_a[0: i+1]))
        action = np.searchsorted(prob, select)
        self.random_counter += 1
        # print(f'{select}+select')
        # print(prob)
        # print(action)
        return action



    def train(self, iter):
        accumulated_reward = []
        steps_per_episode = []
        self.Q[11,:]=0
        reward = 0
        count=0
        for itr in range(iter):
            episode_reward = 0
            episode_steps = 0
            state = 0
            self.world.resetAgent(state)
            # choose action
            a = self.softmaxSelection(state)
            #a = self.softmaxSelection(state)
            expisode = True
            while expisode:
                # perform action
                self.world.move(a)
                # look for reward
                reward = self.world.getReward()
                state_new = int(self.world.getState())

                # new action
                # a_new = self.actionSelection(state_new)
                a_new = self.softmaxSelection(state_new)
                self.update(state,state_new,reward,a,a_new)
                state = state_new
                a = a_new
                #episode record
                episode_reward += reward
                episode_steps += 1

                if reward == 1.0:
                    expisode = False
                    count += 1

            accumulated_reward.append(episode_reward)
            steps_per_episode.append(episode_steps)
            #if(episode_reward==-2.0):
            #    print("episode#"+str(count))
        return accumulated_reward, steps_per_episode

    def Qlearningtrain(self, iter):
        accumulated_reward = []
        steps_per_episode = []
        #initializing
        self.Q = np.loadtxt("initial_Q_values.txt")
        self.Q[11,:]=0
        reward = 0
        count=0
        for itr in range(iter):
            episode_reward = 0
            episode_steps = 0
            #initialize S
            state = 0
            self.world.resetAgent(state)
            expisode = True
            while expisode:
                #action selection
                a = self.softmaxSelection(state)
                # perform action
                self.world.move(a)
                # look for reward
                reward = self.world.getReward()
                state_new = int(self.world.getState())
                #update
                self.Q[state,a]=self.Q[state,a]+(self.alpha * (reward + (self.gamma*np.max(self.Q[state_new, :]))-self.Q[state,a]))
                state = state_new
                #episode record
                episode_reward += reward
                episode_steps += 1

                if reward == 1.0:
                    expisode = False
                    count += 1

            accumulated_reward.append(episode_reward)
            steps_per_episode.append(episode_steps)
            #if(episode_reward==-2.0):
            #    print("episode#"+str(count))
        return accumulated_reward, steps_per_episode



    def plotvalue(self):
        plt.figure()
        plt.plot(range(1000), accumulated_reward)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Reward per Episode")
        plt.show()

        plt.figure()
        plt.plot(range(1000), steps_per_episode)
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        plt.title("Steps per Episode")
        plt.show()


if __name__ == "__main__":
    world = World(3,4)
    world.setReward(2, 3, 1.0) #Goal state
    world.setReward(1, 1, -1.0) #Fear region

    learner = Agent(world)
    #accumulated_reward, steps_per_episode =  learner.train(1000)
    accumulated_reward, steps_per_episode =  learner.Qlearningtrain(1000)

    learner.plotvalue()

    #learner.plotQValues()
