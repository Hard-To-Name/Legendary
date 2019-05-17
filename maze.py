from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import MalmoPython

ACTIONS = ['movenorth 1', 'movesouth 1', 'movewest 1', 'moveeast 1']

MAP_LENGTH = 20
MAP_WIDTH = 20

UNKNOWN = 0
AIR = 1
LAND = 2
TARGET = 3
SELF = 4

epsilon = 0.1
gamma = 0.5


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
                    <SizeAndPosition width="''' + str(MAP_WIDTH) + '''" length="''' + str(MAP_LENGTH) + '''" height="10" xOrigin="-32" yOrigin="69" zOrigin="-5"/>
                    <StartBlock type="emerald_block" fixedToEdge="true"/>
                    <EndBlock type="redstone_block" fixedToEdge="true"/>
                    <PathBlock type="diamond_block"/>
                    <FloorBlock type="air"/>
                    <GapBlock type="air"/>
                    <AllowDiagonalMovement>false</AllowDiagonalMovement>
                  </MazeDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="10000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>CS175AwesomeMazeBot</Name>
                <AgentStart>
                    <Placement x="0.5" y="56.0" z="0.5" yaw="0"/>
                </AgentStart>
                <AgentHandlers>
                    <DiscreteMovementCommands/>
                    <RewardForSendingCommand reward="-1"/>
                    <RewardForTouchingBlockType>
                      <Block reward="100" type="redstone_block" behaviour="onceOnly"/>
                    </RewardForTouchingBlockType>
                    <AgentQuitFromTouchingBlockType>
                        <Block type="redstone_block"/>
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


def build_model(lr=0.001):
    global ACTIONS
    model = Sequential()
    model.add(Dense((MAP_LENGTH, MAP_WIDTH), input_shape=((MAP_LENGTH, MAP_WIDTH),)))
    model.add(PReLU())
    model.add(Dense((MAP_LENGTH, MAP_WIDTH)))
    model.add(PReLU())
    model.add(Dense(len(ACTIONS)))
    model.compile(optimizer='adam', loss='mse')
    return model


class iKun(object):

    def __init__(self, model, memory_length=20, gamma=0.95, epsilon=0.2):
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

        env_size = self.memory[0][0].shape[1]
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

        self.model.fit(inputs, targets, epochs=8, batch_size=10)
        print("loss:", self.model.evaluate(inputs, targets))


class Maze(object):

    def __init__(self, agent_host, agent):
        self.agent_host = agent_host
        self.maze = np.zeros((MAP_LENGTH, MAP_WIDTH))
        self.position, self.target = self.get_position_target()
        self.agent = agent
        self.reward = 0

    def get_position_target(self):
        global MAP_LENGTH, MAP_WIDTH
        world_state = self.agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state <= 0:
            return -1

        msg = world_state.observations[-1].text
        ob = json.loads(msg)
        grid = ob.get(u'floorAll', 0)  # 1D list

        position = [-1, -1]
        target = [-1, -1]
        for i in range(len(grid)):
            if grid[i] == "redstone_block":
                target = [i // MAP_WIDTH, i % MAP_WIDTH]
            elif grid[i] == "emerald_block":
                position = [i // MAP_WIDTH, i % MAP_WIDTH]

        return position, target

    def get_canvas(self):
        global SELF, TARGET
        canvas = np.copy(self.maze)
        canvas[self.position[0]][self.position[1]] = SELF
        canvas[self.target[0]][self.target[1]] = TARGET
        return canvas

    def run(self):
        global ACTIONS, LAND
        canvas = self.get_canvas()

        while True:
            prev_canvas = canvas
            reward = 0
            rnd = random.random()

            if rnd < self.agent.epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(self.agent.predict(canvas))

            agent_host.sendCommand(ACTIONS[action])
            world_state = self.agent_host.getWorldState()
            game_over = False if world_state.is_mission_running else True

            # Update recognized maze
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                print(ob)
                if "Reward" in ob:
                    current_reward = ob["Reward"]
                    if current_reward > self.reward:  # 100 for reaching target
                        self.maze[self.position[0]][self.position[1]] = TARGET
                    elif current_reward + 1 == self.reward:  # -1 for each step
                        self.maze[self.position[0]][self.position[1]] = LAND
                    elif game_over:  # -20 for falling
                        self.maze[self.position[0]][self.position[1]] = AIR
                        current_reward = self.reward - 20
                    reward = current_reward - self.reward
                    self.reward = current_reward

            status = [prev_canvas, action, reward, canvas, game_over]
            self.agent.memorize(status)
            self.agent.train()


if __name__ == '__main__':
    my_client_pool = MalmoPython.ClientPool()
    my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
    mission_xml = GetMissionXML()
    agent_host = MalmoPython.AgentHost()
    agent_host.setDebugOutput(False)

    model = build_model()
    ikun = iKun(model)

    num_reps = 10
    for iRepeat in range(num_reps):
        mission = MalmoPython.MissionSpec(mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()
        agent_host.startMission(mission, my_client_pool, mission_record, 0, "iKun")

        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            maze = Maze(agent_host, ikun)
            maze.run()

    time.sleep(10000)
