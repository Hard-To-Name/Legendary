import time
import threading
import MalmoPython
import random
from collections import defaultdict

def drawMobs(num_mobs):
    xml = ""
    for i in range(num_mobs):
        x = str(random.randint(-17, 17))
        z = str(random.randint(-17, 17))
        xml += '<DrawEntity x="' + x + '" y="214" z="' + z + '" type="Zombie"/>'
    return xml


def drawItems(num_items):
    xml = ""
    for i in range(num_items):
        x = str(random.randint(-17, 17))
        z = str(random.randint(-17, 17))
        xml += '<DrawItem x="' + x + '" y="224" z="' + z + '" type="apple"/>'
    return xml


def getXML(num_mobs = 0, num_items = 4, agent_name="iKun"):
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
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <RewardForCollectingItem>
                    <Item type="apple" reward="10"/>
                  </RewardForCollectingItem>
                  <RewardForTimeTaken initialReward="0" delta="1" density="PER_TICK"/>
                  <RewardForDamagingEntity>
                    <Mob type="Zombie" reward="1"/>
                  </RewardForDamagingEntity>
                  <RewardForSendingCommand reward="-1"/>
                  <VideoProducer>
                    <Width>512</Width>
                    <Height>512</Height>
                  </VideoProducer>
                  <ObservationFromFullStats/>
                </AgentHandlers>
              </AgentSection>'''

    mission_xml += '''</Mission>'''
    return mission_xml


class iKun(object):

    def __init__(self):
        self.speed = 2
        self.turning = 1
        self.ratio = 1.0
        self.defense = 1.0
        self.dodge = 0.0

        self.is_attacking = False
        self.is_crouching = False
        self.is_turning = False
        self.is_jumping = False

        self.inventory = defaultdict(lambda: 0, {})
        self.actions = ['move', 'turn +', 'turn -', 'jump', 'attack', 'crouch']

    def run(self, agent_host):
        while True:
            action = random.choice(self.actions)
            command = ''

            if action == 'move':
                current_speed = self.ratio * self.speed
                command = action + ' ' + str(current_speed)

            elif 'turn' in action:
                if self.is_turning:
                    command = 'turn 0'
                else:
                    current_turning = self.ratio * self.turning
                    command = action + str(current_turning)

            elif action == 'jump':
                if self.is_jumping:
                    command = 'jump 0'
                else:
                    command = 'jump 1'

            elif action == 'attack':
                if self.is_attacking:
                    command = 'attack 0'
                else:
                    command = 'attack 1'

            elif action == 'crouch':
                if self.is_crouching:
                    command = 'crouch 0'
                else:
                    command = 'crouch 1'

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
            world_state = agent_host.getWorldState()

            ikun.run(agent_host)

    time.sleep(10000)
