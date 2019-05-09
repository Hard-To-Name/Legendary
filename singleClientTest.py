import time
import threading
import MalmoPython
import random

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


def getXML(num_mobs = 0, num_items = 4, agent_names = ['A']):
    mission_name = 'multi_agent'

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
                <DrawBlock x="0" y="226" z="0" type="fence"/>''' + drawMobs(10) + drawItems(10) + '''
              </DrawingDecorator>
              <ServerQuitFromTimeUp description="" timeLimitMs="50000"/>
            </ServerHandlers>
          </ServerSection>'''

    # Add an agent section for each robot. Robots run in survival mode.
    # Give each one a wooden pickaxe for protection...
    for i in range(len(agent_names[:-1])):
        mission_xml += '''<AgentSection mode="Survival">
            <Name>''' + agent_names[i] + '''</Name>
            <AgentStart>
              <Placement x="''' + str(random.randint(-17, 17)) + '''" y="204" z="''' + str(
            random.randint(-17, 17)) + '''"/>
              <Inventory>
                <InventoryObject type="iron_axe" slot="0" quantity="1"/>
                <InventoryItem type="planks" variant="acacia"/>
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

    # Add a section for the observer. Observer runs in creative mode.
    mission_xml += '''<AgentSection mode="Creative">
        <Name>''' + agent_names[-1] + '''</Name>
        <AgentStart>
          <Placement x="0.5" y="228" z="0.5" pitch="90"/>
        </AgentStart>
        <AgentHandlers>
          <VideoProducer>
            <Width>512</Width>
            <Height>512</Height>
          </VideoProducer>
          <ObservationFromFullStats/>
        </AgentHandlers>
      </AgentSection>'''

    mission_xml += '''</Mission>'''
    return mission_xml

class ThreadedAgent(threading.Thread):
    def __init__(self, role, clientPool, missionXML):
        threading.Thread.__init__(self)
        print("Initialize thread %s" % role)
        self.role = role
        self.client_pool = clientPool
        self.mission_xml = missionXML
        self.agent_host = MalmoPython.AgentHost()
        self.agent_host.setDebugOutput(False)
        self.mission = MalmoPython.MissionSpec(missionXML, True)
        self.mission_record = MalmoPython.MissionRecordSpec()
        self.actions = ['turn -0.5', 'move 1', 'jump 1']
        self.reward = 0
        self.mission_end_message = ""
        print("Finish initialization.")

    def run(self):
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Attempt to start the mission:
                self.agent_host.startMission(self.mission, self.client_pool, self.mission_record, self.role, "WTF")
                print("Mission started.")
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission",e)
                    print("Is the game running?")
                    exit(1)
                else:
                    time.sleep(1)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        self.runMissionLoop()

    def runMissionLoop(self):
        turn_key = ""
        times = 100
        while True:
            times -= 1
            if times < 0: break
            world_state = self.agent_host.getWorldState()
            if not world_state.is_mission_running:
                break

            self.agent_host.sendCommand(random.choice(self.actions))

        # Close our agent hosts:
        time.sleep(2)
        self.agent_host = None

if __name__ == '__main__':
    my_client_pool = MalmoPython.ClientPool()
    my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
    mission_xml = getXML()
    agent = ThreadedAgent(0, my_client_pool, mission_xml)
    agent.start()
    agent.join()
    time.sleep(10000)
