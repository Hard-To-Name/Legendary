import time
import threading
import MalmoPython
import random
import json
from collections import defaultdict
import tensorflow as tf


ARENA_WIDTH = 60
ARENA_BREADTH = 60


def drawMobs(num_mobs):
    xml = ""
    for i in range(num_mobs):
        x = str(random.randint(-17, 17))
        z = str(random.randint(-17, 17))
        xml += '<DrawEntity x="' + x + '" y="214" z="' + z + '" type="Zombie" />'
    return xml


def drawItems(num_items):
    xml = ""
    for i in range(num_items):
        x = str(random.randint(-17, 17))
        z = str(random.randint(-17, 17))
        xml += '<DrawItem x="' + x + '" y="224" z="' + z + '" type="apple"/>'
    return xml


def getXML(num_mobs = 1, num_items = 4, agent_name="iKun"):
    mission_name = 'iKun'

    mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <About>
            <Summary/>
          </About>
          <ModSettings>
            <MsPerTick>''' + str(50) + '''</MsPerTick>
          </ModSettings>
          <ServerSection>
            <ServerInitialConditions>
              <Time>
                <StartTime>13000</StartTime>
              </Time>
            </ServerInitialConditions>
            <ServerHandlers>
              <FlatWorldGenerator generatorString="3;2*4,225*22;1;" seed=""/>
              <DrawingDecorator>
                <DrawCuboid x1="-20" y1="200" z1="-20" x2="20" y2="200" z2="20" type="glowstone"/>
                <DrawCuboid x1="-19" y1="200" z1="-19" x2="19" y2="227" z2="19" type="stained_glass" colour="RED"/>
                <DrawCuboid x1="-18" y1="202" z1="-18" x2="18" y2="247" z2="18" type="air"/>
                <DrawBlock x="0" y="226" z="0" type="fence"/>''' + drawMobs(num_mobs) + drawItems(num_items) + '''
              </DrawingDecorator>
              <ServerQuitFromTimeUp description="" timeLimitMs="50000"/>
            </ServerHandlers>
          </ServerSection>'''

    # Agent section
    mission_xml += '''<AgentSection mode="Survival">
                <Name>''' + agent_name + '''</Name>
                <AgentStart>
                  <Placement x="''' + str(random.randint(-17, 17)) + '''" y="204" z="''' + str(
        random.randint(-17, 17)) + '''"/>
                <Inventory>
                  <InventoryObject type="iron_axe" slot="0" quantity="1"/>
                </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ChatCommands/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="'''+str(ARENA_WIDTH)+'''" yrange="2" zrange="'''+str(ARENA_BREADTH)+'''" />
                  </ObservationFromNearbyEntities>
                  <VideoProducer>
                    <Width>512</Width>
                    <Height>512</Height>
                  </VideoProducer>
                  <ObservationFromFullStats/>
                </AgentHandlers>
              </AgentSection>'''

    mission_xml += '''</Mission>'''
    return mission_xml


class Model:

    def __init__(self, actions):
        pass

    def predict(self):
        pass

    def fit(self):
        pass


class iKun(object):

    def __init__(self, n = 1, alpha = 0.3, gamma = 1, model = None):
        self.epsilon = 0.5
        self.q_table = {}
        self.n, self.alpha, self.gamma = n, alpha, gamma

        self.location = [0, 0]
        self.life = 0
        self.speed = 2
        self.turning = 1
        self.ratio = 1.0
        self.defense = 1.0
        self.dodge = 0.0
        self.damage_dealt = 0
        self.reward = 0

        self.is_attacking = False
        self.is_crouching = False
        self.is_turning = False
        self.is_jumping = False

        self.inventory = defaultdict(lambda: 0, {})
        self.state = {}
        self.actions = ['move', 'turn +', 'turn -', 'jump', 'attack', 'crouch']

    def run(self, agent_host):

        world_state = agent_host.getWorldState()

        while world_state.is_mission_running:

            zombie_locations = set()

            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                print(ob)

                if "Life" in ob:
                    current_life = ob[u'Life']
                    if current_life < self.life:
                        agent_host.sendCommand("chat MoAiLaoZi!")
                        self.reward -= 20
                    self.life = current_life

                if 'DamageDealt' in ob:
                    current_damage_dealt = ob[u'DamageDealt']
                    if self.damage_dealt < current_damage_dealt:
                        agent_host.sendCommand("chat Ji Ni Tai Mei")
                        self.reward += 5
                    self.damage_dealt = current_damage_dealt

                if "entities" in ob:
                    entities = ob["entities"]
                    for ent in entities:
                        if ent["name"] == "Zombie":
                            zombie_locations.add(tuple([ent["x"], ent["z"]]))
                        elif ent["name"] == "iKun":
                            self.location[0] = ent["x"]
                            self.location[1] = ent["z"]

            action = random.choice(self.actions)
            command = ''

            if action == 'move':
                current_speed = self.ratio * self.speed
                command = action + ' ' + str(current_speed)

            elif 'turn' in action:
                if self.is_turning:
                    command = 'turn 0'
                    self.is_turning = False
                else:
                    current_turning = self.ratio * self.turning
                    command = action + str(current_turning)
                    self.is_turning = True

            elif action == 'jump':
                if self.is_jumping:
                    command = 'jump 0'
                    self.is_jumping = False
                else:
                    command = 'jump 1'
                    self.is_jumping = True

            elif action == 'attack':
                if self.is_attacking:
                    command = 'attack 0'
                    self.is_attacking = False
                else:
                    command = 'attack 1'
                    self.is_attacking = True

            elif action == 'crouch':
                if self.is_crouching:
                    command = 'crouch 0'
                    self.is_crouching = False
                else:
                    command = 'crouch 1'
                    self.is_crouching = True

            print("Command:", command)
            agent_host.sendCommand(command)
            time.sleep(0.1)



if __name__ == '__main__':
    my_client_pool = MalmoPython.ClientPool()
    my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
    mission_xml = getXML()
    agent_host = MalmoPython.AgentHost()
    agent_host.setDebugOutput(False)

    ikun = iKun()

    num_reps = 1
    for iRepeat in range(num_reps):
        mission = MalmoPython.MissionSpec(mission_xml, True)
        mission_record = MalmoPython.MissionRecordSpec()
        agent_host.startMission(mission, my_client_pool, mission_record, 0, "iKun")

        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            ikun.run(agent_host)

    time.sleep(10000)
