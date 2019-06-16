from __future__ import print_function
import os
import sys
import time
import datetime
import json
import random
import numpy as np
import math
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from PIL import Image
from collections import deque
import matplotlib.pyplot as plt
import MalmoPython


ACTIONS = ['movenorth 1', 'movesouth 1', 'movewest 1', 'moveeast 1']
MOVES = {0: [1, -1], 1: [1, 1], 2: [0, -1], 3: [0, 1]}

SIZE = 10
MAP_LENGTH = 21
MAP_WIDTH = 21

UNKNOWN = 0
AIR = 1
LAND = 2
TARGET = 3
SELF = 4

def GetMissionXML(i):
    global SIZE
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
              <About>
                <Summary> Running Maze ''' + str(i) + ''' </Summary>
              </About>
            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"/>
                  <DrawingDecorator>
                    <DrawSphere x="-27" y="70" z="0" radius="30" type="air"/>
                  </DrawingDecorator>
                  <MazeDecorator>
                    <Seed>0</Seed>
                    <SizeAndPosition width="''' + str(SIZE) + '''" length="''' + str(SIZE) + '''" height="10" xOrigin="-32" yOrigin="69" zOrigin="-5"/>
                    <StartBlock type="emerald_block" fixedToEdge="true"/>
                    <EndBlock type="redstone_block" fixedToEdge="true"/>
                    <PathBlock type="diamond_block"/>
                    <FloorBlock type="air"/>
                    <GapBlock type="stone"/>
                    <GapProbability>''' + str(0.5) + '''</GapProbability>
                    <AllowDiagonalMovement>false</AllowDiagonalMovement>
                  </MazeDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="10000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              <AgentSection mode="Survival">
                <Name>CS175AwesomeMazeBot</Name>
                <AgentStart>
                    <Placement x="''' + str(0.5) + '''" y="56.0" z="''' + str(0.5) + '''" yaw="0"/>
                </AgentStart>
                <AgentHandlers>
                    <DiscreteMovementCommands/>
                    <AbsoluteMovementCommands/>
                    <RewardForSendingCommand reward="-1"/>
                    <RewardForTouchingBlockType>
                      <Block reward="100" type="redstone_block" behaviour="onceOnly"/>
                    </RewardForTouchingBlockType>
                    <AgentQuitFromTouchingBlockType>
                        <Block type="redstone_block"/>
                        <Block type="stone"/>
                    </AgentQuitFromTouchingBlockType>
                    <ObservationFromGrid>
                      <Grid name="floorAll">
                        <min x="-10" y="-1" z="-10"/>
                        <max x="10" y="-1" z="10"/>
                      </Grid>
                  </ObservationFromGrid>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''


class Tabular(object):
    def __init__(self, epsilon=0, alpha=0.3, gamma=0.6, q_table={}):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.positive_n = 5
        self.negative_n = 1
        self.q_table = q_table
        self.position = [-1, -1]
        self.boundary = [-1, -1, -1, -1]
        self.maze = np.zeros((MAP_LENGTH, MAP_WIDTH))
        self.exploration_scores = []
        self.rewards = []
        self.nums_steps = []

    def initialize(self,  agent_host):
        global MAP_LENGTH, MAP_WIDTH, TARGET, AIR, LAND, INIT_POS
        grid = -1
        world_state = agent_host.getWorldState()
        while world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                grid = observations.get(u'floorAll', 0)
                break

        if not grid == -1:
            for i in range(len(grid)):
                if grid[i] == "redstone_block":
                    self.maze[i % MAP_WIDTH][i // MAP_WIDTH] = TARGET
                elif grid[i] == "stone":
                    self.maze[i % MAP_WIDTH][i // MAP_WIDTH] = AIR
                elif grid[i] == "diamond_block":
                    self.maze[i % MAP_WIDTH][i // MAP_WIDTH] = LAND
                elif grid[i] == "emerald_block":
                    INIT_POS = [i % MAP_WIDTH, i // MAP_WIDTH]
                    self.position = INIT_POS
                    self.maze[i % MAP_WIDTH][i // MAP_WIDTH] = LAND
                else:
                    continue
                # Track first block
                if self.boundary[0] == -1:
                    self.boundary[0] = i % MAP_WIDTH
                    self.boundary[1] = i // MAP_WIDTH
                # Track last block
                self.boundary[2] = i % MAP_WIDTH
                self.boundary[3] = i // MAP_WIDTH
            print(self.boundary)

    def get_possible_actions(self, agent_host):
        actions = [0, 1, 2, 3]
        if self.position[1] == self.boundary[1]:
            actions.remove(0)
        if self.position[1] == self.boundary[3]:
            actions.remove(1)
        if self.position[0] == self.boundary[0]:
            actions.remove(2)
        if self.position[0] == self.boundary[2]:
            actions.remove(3)
        return actions

    def show(self):
        plt.grid(True)
        nrows, ncols = self.maze.shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = np.copy(self.maze)
        canvas[self.position[0]][self.position[1]] = SELF
        plt.imshow(canvas.transpose(), interpolation='none', cmap='gray')
        plt.draw()
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.pause(0.01)

    def choose_action(self, curr_state, possible_actions):
        if curr_state not in self.q_table:
            self.q_table[curr_state] = {}
        for action in possible_actions:
            if action not in self.q_table[curr_state]:
                self.q_table[curr_state][action] = 0

        rnd = random.random()
        if rnd < self.epsilon:
            a_idx = random.randint(0, len(possible_actions) - 1)
            a = possible_actions[a_idx]
        else:
            max_q = max(self.q_table[curr_state][act] for act in possible_actions)
            max_a = []
            for act in possible_actions:
                if self.q_table[curr_state][act] == max_q:
                    max_a.append(act)
            a = random.choice(max_a)

        return a

    def act(self, agent_host, action):
        global ACTIONS, MOVES, TARGET
        agent_host.sendCommand(ACTIONS[action])
        self.position[MOVES[action][0]] += MOVES[action][1]

        world_state = agent_host.getWorldState()
        game_over = False

        reward = -2
        if self.maze[self.position[0]][self.position[1]] == TARGET:
            reward = 100
            game_over = True
        elif self.maze[self.position[0]][self.position[1]] == AIR:
            reward = -50
            game_over = True

        if game_over:
            while world_state.is_mission_running:
                world_state = agent_host.getWorldState()

        return [game_over, reward]

    def update_q_table(self, tau, S, A, R, T):
        curr_s, curr_a, curr_r = S.popleft(), A.popleft(), R.popleft()
        G = sum([self.gamma ** i * R[i] for i in range(len(S))])

        n = self.positive_n if R[-1] > 0 else self.negative_n
        if tau + n < T:
            G += self.gamma ** n * self.q_table[S[-1]][A[-1]]

        old_q = self.q_table[curr_s][curr_a]
        self.q_table[curr_s][curr_a] = old_q + self.alpha * (G - old_q)


    def run(self, agent_host):
        # plt.ion()
        # plt.show()
        S, A, R = deque(), deque(), deque()
        visited = np.zeros((MAP_LENGTH, MAP_WIDTH))
        done_update = False
        self.initialize(agent_host)

        while not done_update:
            s0 = str(self.position)
            possible_actions = self.get_possible_actions(agent_host)
            a0 = self.choose_action(s0, possible_actions)

            S.append(s0)
            A.append(a0)
            R.append(0)

            T = sys.maxsize
            for t in range(sys.maxsize):
                time.sleep(0.1)
                if t < T:
                    game_over, current_reward = self.act(agent_host, A[-1])
                    if game_over:
                        exploration_score = np.sum(visited) / (SIZE * SIZE)
                        self.exploration_scores.append(exploration_score)
                        current_reward += exploration_score * 70
                        self.rewards.append(current_reward)
                        self.nums_steps.append(t + 1)
                        print("Reward:", current_reward)
                    else:
                        visited[self.position[0]][self.position[1]] = 1
                    # if iRepeat > 100:
                        # self.show()
                    R.append(current_reward)

                    if game_over:
                        T = t + 1
                        S.append('Term State')
                    else:
                        s = str(self.position)
                        S.append(s)
                        possible_actions = self.get_possible_actions(agent_host)
                        next_a = self.choose_action(s, possible_actions)
                        A.append(next_a)

                tau = t - self.negative_n + 1
                if tau >= 0:
                    self.update_q_table(tau, S, A, R, T)

                if tau == T - 1:
                    while len(S) > 1:
                        tau = tau + 1
                        self.update_q_table(tau, S, A, R, T)
                    done_update = True
                    break


if __name__ == '__main__':
    agent_host = MalmoPython.AgentHost()
    agent_host.setDebugOutput(False)
    load_weight_filename = "table.json"
    save_weight_filename = "table.json"

    # Load Q Table
    table = {}
    try:
        with open(load_weight_filename, 'r') as infile:
            table = json.load(infile)
            print("Q table loaded:", table)
    except:
        print("No Q table found. Use empty Q table.")
    tabular = Tabular(q_table=table)

    num_reps = 150
    num_reps_to_save_weights = 50
    for iRepeat in range(num_reps):
        mission_xml = GetMissionXML(iRepeat)
        mission = MalmoPython.MissionSpec(mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()

        my_client_pool = MalmoPython.ClientPool()
        my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

        max_retries = 3
        for retry in range(max_retries):
            try:
                agent_host.startMission(
                    mission, my_client_pool, mission_record, 0, "Tabular")
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission", (iRepeat + 1), ":", e)
                    exit(1)
                else:
                    time.sleep(0.1)

        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)

        tabular.run(agent_host)

        # Save weights
        with open(save_weight_filename, 'w') as outfile:
            json.dump(tabular.q_table, outfile)
        print("Q Table saved.")

        if iRepeat % num_reps_to_save_weights == 0:
            tabular.epsilon -= 0.05
            tabular.epsilon = max(0, tabular.epsilon)

        time.sleep(0.1)

    plt.plot(tabular.exploration_scores)
    plt.xlabel("Number of Epochs")
    plt.ylabel('Exploration Rate per Epoch')
    plt.show()

    plt.plot(tabular.rewards)
    plt.xlabel("Number of Epochs")
    plt.ylabel('Rewards per Epoch')
    plt.show()

    plt.plot(tabular.nums_steps)
    plt.xlabel("Number of Epochs")
    plt.ylabel('Number of Steps Taken per Epoch')
    plt.show()
