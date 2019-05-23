from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
import math
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import MalmoPython

ACTIONS = ['movenorth 1', 'movesouth 1', 'movewest 1', 'moveeast 1']
MOVES = {0: [0, -1], 1: [0, 1], 2: [1, -1], 3: [1, 1]}

SIZE = 10
MAP_LENGTH = 2 * SIZE + 1
MAP_WIDTH = 2 * SIZE + 1

UNKNOWN = 0
AIR = 1
LAND = 2
TARGET = 3
SELF = 4

INIT_POS = [-1, -1]


def GetMissionXML():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
              <About>
                <Summary>Hello world!</Summary>
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
                    <SizeAndPosition width="''' + str(10) + '''" length="''' + str(10) + '''" height="10" xOrigin="-32" yOrigin="69" zOrigin="-5"/>
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


def build_model(load_weight_filename, lr=0.001):
    global ACTIONS
    model = Sequential()
    model.add(Dense(MAP_LENGTH * MAP_WIDTH, input_shape=(MAP_LENGTH * MAP_WIDTH, )))
    model.add(PReLU())
    model.add(Dense(MAP_LENGTH * MAP_WIDTH))
    model.add(PReLU())
    model.add(Dense(len(ACTIONS)))
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
        return self.model.predict(canvas)[0]

    def train(self, batch_size=5):
        global ACTIONS

        env_size = MAP_LENGTH * MAP_WIDTH
        mem_size = len(self.memory)
        data_size = min(mem_size, batch_size)

        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, len(ACTIONS)))

        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace = False)):
            canvas, action, reward, canvas_next, game_over = self.memory[j]
            inputs[i] = canvas
            targets[i] = self.predict(canvas)
            Q_sa = np.max(self.predict(canvas_next))
            if game_over:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * Q_sa

        self.model.fit(inputs, targets, epochs=2, batch_size=5, verbose=0)
        # print("loss:", self.model.evaluate(inputs, targets))


class Maze(object):

    def __init__(self, agent_host, agent):
        self.agent_host = agent_host
        self.maze = np.zeros((MAP_LENGTH, MAP_WIDTH))
        self.visited = np.zeros((MAP_LENGTH, MAP_WIDTH))
        self.position = [-1, -1]
        self.boundary = [-1, -1, -1, -1]
        self.agent = agent
        self.reward = 0

    def initialize(self):
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
                    self.maze[i // MAP_WIDTH][i % MAP_WIDTH] = TARGET
                elif grid[i] == "stone":
                    self.maze[i // MAP_WIDTH][i % MAP_WIDTH] = AIR
                elif grid[i] == "diamond_block":
                    self.maze[i // MAP_WIDTH][i % MAP_WIDTH] = LAND
                elif grid[i] == "emerald_block":
                    INIT_POS = [i // MAP_WIDTH, i % MAP_WIDTH]
                    self.position = INIT_POS
                    self.maze[i // MAP_WIDTH][i % MAP_WIDTH] = LAND
                    print("self.position:", self.position)
                else: continue
                # Track first block
                if self.boundary[0] == -1:
                    self.boundary[0] = i // MAP_WIDTH
                    self.boundary[1] = i % MAP_WIDTH
                # Track last block
                self.boundary[2] = i // MAP_WIDTH
                self.boundary[3] = i % MAP_WIDTH
            print(self.boundary)

    def get_canvas(self):
        global SELF, TARGET
        canvas = np.copy(self.maze)
        canvas[self.position[0]][self.position[1]] = SELF
        return canvas.reshape((1, -1))

    def teleport(self, teleport_x, teleport_z):
        """Directly teleport to a specific position."""
        tp_command = "tp " + str(teleport_x) + " 56 " + str(teleport_z)
        self.agent_host.sendCommand(tp_command)
        good_frame = False
        while not good_frame:
            world_state = self.agent_host.getWorldState()
            if not world_state.is_mission_running:
                print("Mission ended prematurely - error.")
                exit(1)
            if not good_frame and world_state.number_of_video_frames_since_last_state > 0:
                frame_x = world_state.video_frames[-1].xPos
                frame_z = world_state.video_frames[-1].zPos
                if math.fabs(frame_x - teleport_x) < 0.001 and math.fabs(frame_z - teleport_z) < 0.001:
                    good_frame = True

    def run(self):
        global ACTIONS, LAND
        self.initialize()
        canvas = self.get_canvas()

        while True:
            prev_canvas = canvas
            reward = 0
            rnd = random.random()

            if rnd < self.agent.epsilon:
                action = random.randint(0, 3)
            else:
                actions = self.agent.predict(canvas)
                action = np.argmax(actions)

                while action == 0 and self.position[0] == self.boundary[0] or\
                        action == 1 and self.position[0] == self.boundary[2] or \
                        action == 2 and self.position[1] == self.boundary[1] or\
                        action == 3 and self.position[1] == self.boundary[3]:
                    actions[action] = -1e6
                    action = np.argmax(actions)

            print("action:", action)
            self.position[MOVES[action][0]] += MOVES[action][1]
            print("position:", self.position)
            agent_host.sendCommand(ACTIONS[action])
            time.sleep(0.3)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            game_over = False if world_state.is_mission_running else True

            # Update recognized maze
            current_reward = 0
            if len(world_state.rewards) > 0:
                current_reward = world_state.rewards[-1].getValue()

            if self.visited[self.position[0]][self.position[1]] == 0:
                current_reward += 2
                self.visited[self.position[0]][self.position[1]] = 1

            if game_over and current_reward < 0:  # -20 for falling
                current_reward = -20

            print("current_reward:", current_reward)
            self.reward += current_reward

            canvas = self.get_canvas()
            status = [prev_canvas, action, current_reward, canvas, game_over]
            self.agent.memorize(status)
            self.agent.train()

            if game_over:
                print("game over")
                return 0


if __name__ == '__main__':
    mission_xml = GetMissionXML()
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
    for iRepeat in range(num_reps):

        if not iRepeat == 0:
            # self.boundary[0] <= INIT_POS[0] <= self.boundary[2]
            # self.boundary[1] <= INIT_POS[1] <= self.boundary[3]
            startX = random.randint(maze.boundary[0], maze.boundary[2]) - INIT_POS[0]
            startZ = random.randint(maze.boundary[1], maze.boundary[3]) - INIT_POS[1]
            maze.teleport(startX, startZ)

        mission = MalmoPython.MissionSpec(mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()
        my_client_pool = MalmoPython.ClientPool()
        my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

        max_retries = 3
        for retry in range(max_retries):
            try:
                agent_host.startMission(mission, my_client_pool, mission_record, 0, "iKun")
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
        maze.run()

        # Save weights
        if num_reps % num_reps_to_save_weights == 0:
            ikun.model.save_weights(save_weight_filename + '.h5', overwrite=True)
            print("Weights saved.")

    time.sleep(10000)
