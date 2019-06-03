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
MAP_LENGTH = 2 * SIZE + 1
MAP_WIDTH = 2 * SIZE + 1

def GetMissionXML(i):
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
                    <SizeAndPosition width="''' + str(10) + '''" length="''' + str(10) + '''" height="10" xOrigin="0" yOrigin="69" zOrigin="0"/>
                    <StartBlock type="emerald_block" fixedToEdge="true"/>
                    <EndBlock type="redstone_block" fixedToEdge="true"/>
                    <PathBlock type="diamond_block"/>
                    <FloorBlock type="air"/>
                    <GapBlock type="stone"/>
                    <GapProbability>''' + str(0.2) + '''</GapProbability>
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
    def __init__(self, epsilon=0.8, alpha=0.2, gamma=0.6, n=1, q_table={}):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.q_table = q_table
        self.position = [-1, -1]
        self.boundary = [-1, -1, -1, -1]

    def initialize(self, agent_host):
        global MAP_WIDTH
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
                if grid[i] == "emerald_block":
                    self.position = [i % MAP_WIDTH, i // MAP_WIDTH]
                if grid[i] == "air": continue

                if self.boundary[0] == -1:
                    self.boundary[0] = i % MAP_WIDTH
                    self.boundary[1] = i // MAP_WIDTH
                # Track last block
                self.boundary[2] = i % MAP_WIDTH
                self.boundary[3] = i // MAP_WIDTH

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
        global ACTIONS, MOVES
        agent_host.sendCommand(ACTIONS[action])
        self.position[MOVES[action][0]] += MOVES[action][1]

        world_state = agent_host.getWorldState()
        game_over = False if world_state.is_mission_running else True

        reward = 0
        if len(world_state.rewards) > 0:
            reward = world_state.rewards[-1].getValue()

        if game_over and reward < 50:  # -50 for falling
            reward = -50

        return [game_over, reward]

    def update_q_table(self, tau, S, A, R, T):
        curr_s, curr_a, curr_r = S.popleft(), A.popleft(), R.popleft()
        G = sum([self.gamma ** i * R[i] for i in range(len(S))])
        if tau + self.n < T:
            G += self.gamma ** self.n * self.q_table[S[-1]][A[-1]]

        old_q = self.q_table[curr_s][curr_a]
        self.q_table[curr_s][curr_a] = old_q + self.alpha * (G - old_q)


    def run(self, agent_host):
        S, A, R = deque(), deque(), deque()
        present_reward = 0
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

                tau = t - self.n + 1
                if tau >= 0:
                    self.update_q_table(tau, S, A, R, T)

                if tau == T - 1:
                    while len(S) > 1:
                        tau = tau + 1
                        self.update_q_table(tau, S, A, R, T)
                    done_update = True
                    print(self.q_table)
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
    except:
        print("No Q table found. Use empty Q table.")
    tabular = Tabular(q_table=table)

    num_reps = 100000
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
        if iRepeat % num_reps_to_save_weights == 0:
            tabular.epsilon -= 0.05
            with open(save_weight_filename, 'w') as outfile:
                json.dump(tabular.q_table, outfile)
            print("Q Table saved.")

        time.sleep(1)
