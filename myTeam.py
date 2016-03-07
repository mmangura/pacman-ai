# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
# 
# This version edited by Marc Manguray & Michael Cordoza

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import distanceCalculator


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'BaseOffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on). 
    
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    ''' 
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py. 
    '''
    CaptureAgent.registerInitialState(self, gameState)

    ''' 
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)


    ''' 
    You should change this in your own agent.
    '''

    return random.choice(actions)

class ReflexCaptureAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class BaseOffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    #gets actions such as stop and reverse for the attacking agent
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
  
    if action == Directions.STOP: features['Ostop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['Oreverse'] = 1

    #stays away from any ghost when a ghost
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    defenders = [a for a in enemies if not(a.isPacman) and a.getPosition() != None]
    features['numDefenders'] = len(defenders)
    if len(defenders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
      if min(dists) <= 5:
        features['defenderDistance'] = min(dists)

    temp = self.getOpponents(successor)    
    enemyState = successor.getAgentState(temp[0])

    features['enemyScaredTimer'] = enemyState.scaredTimer

    #testing attacking stance
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders1'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance1'] = min(dists)

    #keeps distance away from enemy when scared
    if myState.scaredTimer > 0:
      if features['invaderDistance1'] <= 4:
        features['invaderDistance1'] = -features['invaderDistance1']


    if (features['invaderDistance1'] > 10 and myState.isPacman):
      features['numInvaders1'] = 0
      features['invaderDistance1'] = 0


    # Compute distance to the nearest food
    capsuleList = self.getCapsules(successor)
    if (len(capsuleList) > 0): # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minCapsuleDistance = min([self.getMazeDistance(myPos, capsule) for capsule in reversed(capsuleList)])
      features['capsule'] = minCapsuleDistance
      


    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if (len(foodList) > 0): # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders1': -1000, 'invaderDistance1': -2, 'successorScore': 100, 'distanceToFood': -2, 'capsule': -2, 'defenderDistance': 2, 'Ostop': -100, 'Oreverse': -2, 'enemyScaredTimer': 20}


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.allEnemiesScared = 0
    self.allEnemiesTooFarToDefend = 0
    myPos = gameState.getAgentState(self.index).getPosition()
    foodToDefend = self.getFoodYouAreDefending(gameState).asList() # Make a list of Locations to Patrol

    self.hotspots = util.Counter()
    self.suspiciousLocations = util.Stack()
    self.currentSuspicion = None

    for loc in foodToDefend:
      self.hotspots[loc] = self.getMazeDistance(myPos, loc)
    
    self.hotspots.normalize()

  def chooseAction(self, gameState):
    enemyIndex = self.getOpponents(gameState)
    enemies = [gameState.getAgentState(i) for i in enemyIndex] # List of Enemy Agent States
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None] # List of Enemy Invaders
    myPos = gameState.getAgentState(self.index).getPosition() # Position of Agent

    # Updates the hotspots and currentSuspicion if it is changed
    for hotspot, value in self.hotspots.items():
      if hotspot not in self.getFoodYouAreDefending(gameState).asList():
        del self.hotspots[hotspot]
        self.currentSuspicion = hotspot
        self.hotspots.normalize()

    # Updates current Suspicion if suspicion is relieved
    if self.currentSuspicion is not None:
      if self.getMazeDistance(myPos, self.currentSuspicion) is 0:
        self.currentSuspicion = None

    self.allEnemiesScared = 0
    for e in enemies:
      if e.scaredTimer > 0:
        self.allEnemiesScared = 1

    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    chosenAction = random.choice(bestActions)
    return chosenAction

  def evaluate(self, gameState, action):

    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)

    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    enemyIndex = self.getOpponents(successor)
    enemies = [successor.getAgentState(i) for i in enemyIndex] # List of Enemy Agent States
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None] # List of Observable Invading Agent States
    defenders = [a for a in enemies if not(a.isPacman) and a.getPosition() != None] # List of Observable Defending Agents
    scaredDefenders = [d for d in defenders if d.scaredTimer > 1] # List of Defending Agents taht are scared
    foodToDefendLocations = self.getFoodYouAreDefending(successor).asList() # Locations of Team Food


    ######################
    # UNIVERSAL FEATURES # 'onDefense', 'stop', 'reverse', 'succesorScore'
    ######################
    ## (1) When Agent is on Defense Side (0) When on the Offense Side
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    ## (1) When action is STOP (0) if not
    if action == Directions.STOP: features['stop'] = 1 
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction] 
    if action == rev: features['reverse'] = 1

    ## The score if an action is taken
    features['successorScore'] = self.getScore(successor)

    ###################
    # DEFEND FEATURES # 'numInvaders', 'invaderDistance'
    ###################
    ## Number of invaders seen by agents
    features['numInvaders'] = len(invaders) 
    
    ## Computes distance of closest invader that our Agents can see
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    ## Agent is to run from Invader if Scared but don't stray too far to lose sight 
    if myState.scaredTimer > 0:
        if features['invaderDistance'] <= 4:
            features['invaderDistance'] = -features['invaderDistance']


    ###################
    # PATROL FEATURES # 'patrolDistance', 'suspiciousLocationDistance'
    ###################
    ## Gets the distance from the hottest hotspot
    features['patrolDistance'] = self.getMazeDistance(myPos, self.hotspots.argMax())

    ## Gets the distance of the most recent food that disappeared. 
    if self.currentSuspicion is not None:
        features['suspiciousLocationDistance'] = self.getMazeDistance(myPos, self.currentSuspicion)

    ###################
    # ATTACK FEATURES # 'foodDistance', 'scaredDistance', 'defenderDistance', 'capsuleDistance'
    ###################
    ## Initializes the Attack Phase
    if self.allEnemiesScared and (features['successorScore'] > 0):

      ## Ignores Most Defensive Features
      features['onDefense'] = 0
      features['patrolDistance'] = 0
      
      ## Finds the closest food
      foodList = self.getFood(successor).asList()
      foodDists = [self.getMazeDistance(myPos, f) for f in foodList]
      features['foodDistance'] = min(foodDists)

      ## Finds the closest scared ghost distance
      if len(scaredDefenders) > 0: 
        scaredDefendersDists = [self.getMazeDistance(myPos, d.getPosition()) for d in scaredDefenders]
        features['scaredDistance'] = min(scaredDefendersDists)

      ## Finds the closest defender ghost distance
      if len(defenders) > 0:
        defenderDists = [self.getMazeDistance(myPos, d.getPosition()) for d in defenders]
        features['defenderDistance'] = min(defenderDists)

      ## Finds the closest capsule
      capsuleList = self.getCapsules(successor)
      if len(capsuleList):
        capsuleDists = [self.getMazeDistance(myPos, c) for c in capsuleList]
        features['capsuleDistance'] = min(capsuleDists)

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 
            'onDefense': 100, 
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -4,
            'patrolDistance': -1,
            'foodDistance': -5,
            'successorScore': 100,
            'scaredDistance': -1, 
            'defenderDistance': 1,
            'capsuleDistance': -5,
            'suspiciousLocationDistance': -3}
