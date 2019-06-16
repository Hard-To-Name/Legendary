from __future__ import print_function
import os
import sys
import time
import datetime
import json
import random
import numpy as np
import math
import copy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
import MalmoPython

ACTIONS = ['movenorth 1', 'movesouth 1', 'movewest 1', 'moveeast 1']
MOVES = {0: [1, -1], 1: [1, 1], 2: [0, -1], 3: [0, 1]}

SIZE = 10
MAP_LENGTH = 2 * SIZE + 1
MAP_WIDTH = 2 * SIZE + 1

UNKNOWN = 0
AIR = 1
LAND = 2
TARGET = 3
SELF = 4

INIT_POS = [-1, -1]


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
                      <Block reward="-50" type="stone" behaviour="onceOnly"/>
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


def build_model(load_weight_filename, lr=0.001):
    global ACTIONS
    model = Sequential()
    model.add(Dense(128, input_shape=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    #model.load_weights(load_weight_filename + '.h5')
    model.compile(optimizer='adam', loss='mse')
    return model


class iKun(object):

    def __init__(self, model, memory_length=20, gamma=0.8, epsilon=0):
        self.model = model
        self.memory = []
        self.memory_length = memory_length
        self.gamma = gamma
        self.epsilon = epsilon

    def memorize(self, status):
        self.memory.append(status)
        if len(self.memory) > self.memory_length:
            del self.memory[0]

    def predict(self, canvas):
        return self.model.predict(canvas)

    def train(self, repeat_time, loss_value, batch_size=5):
        global ACTIONS

        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = []
        Y = []
        for i in range(batch_size):
            state, action, reward, next_state, next_movable, done = minibatch[i]
            input_action = [state, action]
            if done:
                target_f = reward
            else:
                next_reward = []
                for a in next_movable:
                    nn_temp = copy.deepcopy(next_state)
                    nn_temp[MOVES[a][0]] += MOVES[a][1]
                    np_next = np.array([[next_state, nn_temp]])
                    next_reward.append(self.predict(np_next))
                r_max = np.max(next_reward)
                target_f = reward + self.gamma * r_max
            X.append(input_action)
            Y.append(target_f)

        npX = np.array(X)
        npY = np.array(Y)

        self.model.fit(npX, npY, epochs=1, batch_size=5, verbose=0)
        # loss_value.append(self.model.evaluate(npX, npY))
        # print("loss:", loss_value[-1])

        if repeat_time > 0 and repeat_time % 20 == 0:
            self.epsilon *= 0.95


class Maze(object):

    def __init__(self, agent_host, agent):
        self.agent_host = agent_host
        self.position = [-1, -1]
        self.boundary = [-1, -1, -1, -1]
        self.visited = np.zeros((MAP_LENGTH, MAP_WIDTH))
        self.size = MAP_LENGTH*MAP_WIDTH
        self.agent = agent
        self.target = [-1, -1]

    def initialize(self):
        global MAP_LENGTH, MAP_WIDTH, TARGET, AIR, LAND, INIT_POS
        self.visited = np.zeros((MAP_LENGTH, MAP_WIDTH))
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
                if grid[i] == 'redstone_block':
                    self.target = [i % MAP_WIDTH, i // MAP_LENGTH]
                if grid[i] == "air":
                    continue

                if self.boundary[0] == -1:
                    self.boundary[0] = i % MAP_WIDTH
                    self.boundary[1] = i // MAP_WIDTH
                # Track last block
                self.boundary[2] = i % MAP_WIDTH
                self.boundary[3] = i // MAP_WIDTH
            x, z = self.position
            self.visited[x-self.boundary[0], z-self.boundary[1]] = 1

    def get_possible_actions(self):
        actions = []
        if self.position[1] > self.boundary[1]:
            actions.append(0)
        if self.position[1] < self.boundary[3]:
            actions.append(1)
        if self.position[0] > self.boundary[0]:
            actions.append(2)
        if self.position[0] < self.boundary[2]:
            actions.append(3)
        return actions

    def teleport(self, teleport_x, teleport_z):
        """Directly teleport to a specific position."""

        tp_command = "tp "+str(teleport_x+0.5)+" 71 "+str(teleport_z+0.5)
        self.agent_host.sendCommand(tp_command)

    def choose_actions(self, state, possible_moves):
        rnd = random.random()
        if self.agent.epsilon > rnd:
            a = random.choice(possible_moves)
            state[MOVES[a][0]] += MOVES[a][1]
            return state, a
        else:
            best = []
            max_v = -np.inf
            for a in possible_moves:
                temp = copy.deepcopy(state)
                temp[MOVES[a][0]] += MOVES[a][1]
                np_action = np.array([[state, temp]])
                act_value = self.agent.predict(np_action)
                if act_value > max_v:
                    best = [(temp, a)]
                    max_v = act_value
                else:
                    best.append((temp, a))
            return random.choice(best)

    def run(self, iRepeat, loss_value):
        global ACTIONS, LAND
        self.initialize()

        if not iRepeat == 0:
            '''
            position is wrong, the maze is at height y=69, so teleport to y=71
            to make sure that agent won't fail. Maze's position is at x=-32 to
            x= -32+length of edge, z=-5 to z=-5+length of edge. Make sure that
            the final position is +-0.5, e.g. x = -22.5, y = 71, z = -0.5
            '''
            target = [self.target[0]-self.boundary[0],
                      self.target[1]-self.boundary[1]]
            if iRepeat < 50:
                startX = random.randint(
                    max(0, target[0]-1), min(SIZE-1, target[0]+1))
                startZ = random.randint(
                    max(0, target[1]-1), min(SIZE-1, target[1]+1))
            elif iRepeat < 100:
                startX = random.randint(
                    max(0, target[0]-2), min(SIZE-1, target[0]+2))
                startZ = random.randint(
                    max(0, target[1]-2), min(SIZE-1, target[1]+2))
            elif iRepeat < 150:
                startX = random.randint(
                    max(0, target[0]-3), min(SIZE-1, target[0]+3))
                startZ = random.randint(
                    max(0, target[1]-3), min(SIZE-1, target[1]+3))
            elif iRepeat < 200:
                startX = random.randint(
                    max(0, target[0]-4), min(SIZE-1, target[0]+4))
                startZ = random.randint(
                    max(0, target[1]-4), min(SIZE-1, target[1]+4))
            else:
                startX, startZ = [
                    self.position[0]-self.boundary[0], self.position[1]-self.boundary[1]]
            maze.teleport(startX, startZ)
            old_pos = self.position
            self.position = [startX + self.boundary[0],
                             startZ + self.boundary[1]]
            time.sleep(0.2)
        while True:
            prev_pos = self.position
            movables = self.get_possible_actions()
            next_state, action = self.choose_actions(self.position, movables)

            print("action:", action)
            self.position = next_state
            print("position:", self.position)
            agent_host.sendCommand(ACTIONS[action])
            time.sleep(0.3)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            game_over = False if world_state.is_mission_running else True

            x, z = self.position
            self.visited[x-self.boundary[0], z-self.boundary[1]] = 1
            # Update recognized maze
            current_reward = 0
            if len(world_state.rewards) > 0:
                current_reward = world_state.rewards[-1].getValue()

            if current_reward > 1:
                time.sleep(2)
            if game_over:
                current_reward += 100 * np.sum(self.visited) / self.size

            print("current_reward:", current_reward)
            next_movable = self.get_possible_actions()

            status = [prev_pos, next_state, current_reward,
                      self.position, next_movable, game_over]
            self.agent.memorize(status)
            self.agent.train(iRepeat, loss_value)

            if game_over:
                print("game over")
                return 0


if __name__ == '__main__':
    agent_host = MalmoPython.AgentHost()
    agent_host.setDebugOutput(False)
    load_weight_filename = "weights"
    save_weight_filename = "weights"

    model = build_model(load_weight_filename)
    # Save model
    with open(save_weight_filename + '.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    ikun = iKun(model)
    maze = Maze(agent_host, ikun)

    num_reps = 100000
    num_reps_to_save_weights = 50
    loss = []
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
                    mission, my_client_pool, mission_record, 0, "iKun")
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

        print("maze run")
        maze.run(iRepeat, loss)

        # Save weights
        if num_reps % num_reps_to_save_weights == 0:
            ikun.model.save_weights(
                save_weight_filename + '.h5', overwrite=True)
            print("Weights saved.")

    time.sleep(10000)
