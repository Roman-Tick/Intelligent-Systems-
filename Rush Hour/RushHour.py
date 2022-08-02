"""
Rush Hour Game


"""
import random
import numpy
import time
from copy import deepcopy
from queue import PriorityQueue

CARS = {'X', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'}
TRUCKS = {'O', 'P', 'Q', 'R'}

Number_of_searched_states = 0

def read_data():
    RH_Input_file = open("rh.txt")
    RH_game_data = []
    read_line = False
    for txt_line in RH_Input_file:
        if "--- RH-input ---" in txt_line:
            read_line = True
            continue
        if "--- end RH-input ---" in txt_line:
            read_line = False
        if read_line == True:
            RH_game_data.append(txt_line[:])

    RH_Input_file.close()
    return RH_game_data

def create_game_map(RH_game_data):
    game_map = []
    x = 0
    for i in range(6):
        row = []
        for j in range(6):
            row.append(RH_game_data[x])
            x = x + 1
        game_map.append(row)

    #print(RH_game_data) #debug
    #print(game_map) #debug
    #print_game_map(game_map) #debug

    return game_map

def print_game_map(map):
    index = 1
    print('    1 2 3 4 5 6')
    print('  +-------------+')
    for i in range(6):
        print(index, "|", end=' ')
        index = index + 1
        for j in range(6):
            print(map[i][j], end=' ')
        if index == 4:
            print("   ==>")
        else:
            print('|')
    print('  +-------------+')
    print('    a b c d e f')

class board():
    def __init__(self, map, vehicle_list, path):
        self.map = map
        self.vehicle_list = vehicle_list
        self.path = path
        self.priority = 100000

    def __eq__(self, other):
        if isinstance(other, board):
            if other.map == self.map:
                return True
        return False

class vehicles():
    def __init__(self, id, length, x, y):
        """
        id: Unique letter identifier for each car
        orientation: Vertical or Horizontal
        x: left/top x cooridnate
        Y: left/top y cooridnate
        front_x: right/bottom x cooridnate
        front_y: right/bottom y cooridnate
        """
        self.id = id
        self.length = length
        
        if 0 <= y <= 5 or 0 <= x <= 5:
            self.x = x
            self.y = y
        else:
            print("invalid x/y (", x, ", ", y, ") for vehicle ", id)
        
        #placeholder values - get updated later
        self.orientation = 'X'
        self.end_x = 0
        self.end_y = 0
    
    def __eq__(self,other):
        if isinstance(other, vehicles):
            if other.id == self.id and other.orientation == self.orientation and other.length == self.length and other.x == self.x and other.y == self.y and other.end_x == self.end_x and other.end_y == self.end_y:
                return True
        return False 
        

def get_vehicle(list, search_id):
    for v in list:
        if v.id == search_id:
            return v
    return False

def update_vehicle_list(list_of_vehicles, square, length, x, y):
    found_v = get_vehicle(list_of_vehicles, square)
    if found_v == False:      #contains(list_of_vehicles, lambda v: v.id == square):
        new_v = vehicles(square, length, x, y)
        list_of_vehicles.append(new_v)
    else:
        found_v.end_x = x
        found_v.end_y = y

        if found_v.x == found_v.end_x:
            found_v.orientation = 'H'
        else:
            found_v.orientation = 'V'

    return list_of_vehicles

def create_vehicles_list(board):
    #if x in CARS or TRUCKS and x not in vehcile_list.id
    x = 0
    y = 0
    list_of_vehicles = []

    for row in board:
        for square in row:
            if square in CARS:
                list_of_vehicles = update_vehicle_list(list_of_vehicles, square, 2, x, y)
            elif square in TRUCKS:
                list_of_vehicles = update_vehicle_list(list_of_vehicles, square, 3, x, y)
            x = x + 1
        y = y + 1
        x = 0

    return list_of_vehicles

def solved_state(current_state): #check if row 2 of map only contains 'X' or '.' to the right of X
    i = 0
    check_from_X = False
    while i < 6:
        if current_state.map[2][i] == 'X':
            check_from_X = True

        if check_from_X == True:
            if current_state.map[2][i] != 'X' and current_state.map[2][i] != '.':
                return False
        i = i + 1
    #add X movement to path
    return True

def update_map(map, new_v, old_v):
    map[old_v.y][old_v.x] = '.'
    map[old_v.end_y][old_v.end_x] = '.'
    if old_v.length == 3:
        old_mid_x = (old_v.x + old_v.end_x)/2
        old_mid_y = (old_v.y + old_v.end_y)/2
        map[int(old_mid_y)][int(old_mid_x)] = '.'
        mid_x = (new_v.x + new_v.end_x)/2
        mid_y = (new_v.y + new_v.end_y)/2
        map[int(mid_y)][int(mid_x)] = new_v.id
    map[new_v.y][new_v.x] = new_v.id
    map[new_v.end_y][new_v.end_x] = new_v.id
    return map

def expand(current_state, BSF_Queue, visited_states):
    """
    outline: 
        -cycle through vehicle list 
        -move car by one if its allowed
        -created new state (with the moved) if it doesnt already exist
    """
    for v in current_state.vehicle_list:
        if v.orientation == 'V':
            move = 1
            possible_move = True
            while possible_move == True:
                if v.x - move >= 0 and current_state.map[v.y][v.x - move] == '.':
                    #print("move left", v.id, ": to (", v.x -1, v.y, ") == ", current_state.map[v.y][v.x -1]) #debug
                    #print("info: ", v.id, v.x, v.y, v.orientation, v.end_x, v.end_y) #debug

                    #create new vehicle
                    new_v = vehicles(v.id, v.length, v.x - move, v.y)
                    new_v.end_x = v.end_x - move
                    new_v.end_y = v.end_y
                    new_v.orientation = v.orientation

                    #create new vehicle list
                    new_v_list = current_state.vehicle_list.copy()
                    new_v_list.remove(v)
                    new_v_list.append(new_v)
                
                    #create new board
                    updated_map = update_map(deepcopy(current_state.map), new_v, v)
                    updated_path = current_state.path.copy()
                    updated_path.append(v.id + 'L' + str(move))
                    new_state = board(updated_map, new_v_list, updated_path) #add path in board init

                    #check if state already exists
                    if state_exists(BSF_Queue, new_state) == False and state_exists(visited_states, new_state) == False:
                        #add steps taken/path
                        BSF_Queue.append(new_state) #add new_state to queue
                else:
                    possible_move = False
                move = move + 1
                    
            move = 0
            possible_move = True
            while possible_move == True:
                if v.x + v.length + move <= 5 and current_state.map[v.y][v.x + v.length + move] == '.':
                    #print("move right", v.id, ": to (", v.x + v.length, v.y, ") == ", current_state.map[v.y][v.x + v.length]) #debug
                    #print("info: ", v.id, v.x, v.y, v.orientation, v.end_x, v.end_y) #debug
                
                    #create new vehicle
                    new_v = vehicles(v.id, v.length, v.x + move + 1, v.y)
                    new_v.end_x = v.end_x + move + 1
                    new_v.end_y = v.end_y
                    new_v.orientation = v.orientation

                    new_v_list = current_state.vehicle_list.copy()
                    new_v_list.remove(v)
                    new_v_list.append(new_v)
                
                    #create new board
                    updated_map = update_map(deepcopy(current_state.map), new_v, v)
                    updated_path = current_state.path.copy()
                    updated_path.append(v.id + 'R' + str(move+1))
                    new_state = board(updated_map, new_v_list, updated_path)
                    #new_state.path = deepcopy(current_state.path)
                    #move = v.id +  'R' + '1'
                    #new_state.path.append(move)
                    #add steps taken/path

                    #check if state already exists
                    if state_exists(BSF_Queue, new_state) == False and state_exists(visited_states, new_state) == False:
                        BSF_Queue.append(new_state) #add new_state to queue
                else:
                    possible_move = False
                move = move + 1

        else: # v.orientation == 'H'
            move = 1
            possible_move = True
            while possible_move == True:
                if v.y - move >= 0 and current_state.map[v.y - move][v.x] == '.':
                    #print("move up", v.id, ": to (", v.x, v.y -1, ") == ", current_state.map[v.y -1][v.x]) #debug
                    #print("info: ", v.id, v.x, v.y, v.orientation, v.end_x, v.end_y) #debug

                    #create new vehicle
                    new_v = vehicles(v.id, v.length, v.x, v.y - move)
                    new_v.end_x = v.end_x
                    new_v.end_y = v.end_y  - move
                    new_v.orientation = v.orientation

                    new_v_list = current_state.vehicle_list.copy()
                    new_v_list.remove(v)
                    new_v_list.append(new_v)
                
                    #create new board
                    updated_map = update_map(deepcopy(current_state.map), new_v, v)
                    updated_path = current_state.path.copy()
                    updated_path.append(v.id + 'U' + str(move))
                    new_state = board(updated_map, new_v_list, updated_path)
                    #add steps taken/path
                
                    #check if state already exists
                    if state_exists(BSF_Queue, new_state) == False and state_exists(visited_states, new_state) == False:
                        BSF_Queue.append(new_state)#add new_state to queue
                else:
                    possible_move = False
                move = move + 1

            move = 0
            possible_move = True
            while possible_move == True:
                if v.y + v.length + move <= 5 and current_state.map[v.y + v.length + move][v.x] == '.':
                    #print("move down", v.id, ": to (", v.x, v.y + v.length, ") == ", current_state.map[v.y + v.length][v.x]) #debug
                    #print("info: ", v.id, v.x, v.y, v.orientation, v.end_x, v.end_y) #debug

                    #create new vehicle
                    new_v = vehicles(v.id, v.length, v.x, v.y + move + 1)
                    new_v.end_x = v.end_x
                    new_v.end_y = v.end_y + move + 1
                    new_v.orientation = v.orientation

                    new_v_list = current_state.vehicle_list.copy()
                    new_v_list.remove(v)
                    new_v_list.append(new_v)
                
                    #create new board
                    updated_map = update_map(deepcopy(current_state.map), new_v, v)
                    updated_path = current_state.path.copy()
                    updated_path.append(v.id + 'D' + str(move+1))
                    new_state = board(updated_map, new_v_list, updated_path)
                    #add steps taken/path
                
                    #check if state already exists
                    if state_exists(BSF_Queue, new_state) == False and state_exists(visited_states, new_state) == False:
                        BSF_Queue.append(new_state)#add new_state to queue
                else:
                    possible_move = False
                move = move + 1
                
    return BSF_Queue

def state_exists(list, new_state): #checks whether a state exists within a certain list
    for state in list:
        if new_state == state:
            return True
    return False

def get_state(list, current_state): #returns state with the same map for a given list
    for state in list:
        if current_state == state:
            return state
    return None

def find_optimal(goal_states):
    opt_sol = goal_states[0].path
    for node in goal_states:
        if len(node.path) < len(opt_sol):
            opt_sol = node.path
    print("Optimal Sol:     ", opt_sol)
    print("goal nodes: ", len(goal_states))

def iterative_deepening_DFS(state, current_depth, max_depth, visited_states):
    global Number_of_searched_states
    Number_of_searched_states = Number_of_searched_states + 1
    if solved_state(state): #found goal state
        print("IDDFS: cur_depth = ", current_depth, " max_depth = ", max_depth)
        return state, True
    
    #child_states = []
    #if current state in visited then skip - dont really need this tho cause it will have no children to expand
    
    visited_states.append(state) #add current state to visited states
    temp_states = expand(state, [], visited_states) #expand state/find children
    child_states = deepcopy(temp_states)

    if current_depth == max_depth:
        if len(child_states) > 0:
            return None, False
        else:
            return None, True
    
    bottom_reached = True
    #x = random.randint(0, len(child_states) - 1)

    for x in range(len(child_states)):
        Goal_state, bottom_reached_rec = iterative_deepening_DFS(child_states[x], current_depth + 1, max_depth, visited_states)
        #print("child")
        if Goal_state is not None: #found goal state going down that child
            return Goal_state, True
        bottom_reached = bottom_reached and bottom_reached_rec #?
        

    #print("went through all children and found no Goal State")
    return None, bottom_reached
        
def Heuristic_calculator(state, Number_of_Heuristics):
    if Number_of_Heuristics == 1:
        return H1(state) + len(state.path)
    if Number_of_Heuristics == 2:
        return H2(state) + len(state.path)
    if Number_of_Heuristics == 3:
        return H3(state) + len(state.path)
    if Number_of_Heuristics== 4:
        return H2(state) + H3(state)
    

def H1(state): # checks X cars distance from exit
    distance = 0
    index = 5
    while state.map[2][index] != 'X':
        index = index - 1
        distance = distance + 1
    return distance

def H2(state): # checks number of cars blocking the exit
    blocking_cars = 0
    index = 5
    while state.map[2][index] != 'X':
        if state.map[2][index] in CARS or state.map[2][index] in TRUCKS:
            blocking_cars = blocking_cars + 1
        index = index - 1
    return blocking_cars

def H3(state): #checks number of cars blocking the cars from H2
    index = 5
    list_of_blocking_cars = []
    while state.map[2][index] != 'X':
        if state.map[2][index] in CARS or state.map[2][index] in TRUCKS:
            list_of_blocking_cars = check_cars_blocking(state.map, get_vehicle(state.vehicle_list, state.map[2][index]), list_of_blocking_cars)
        index = index - 1
    #print("num of blocking v's: ", len(list_of_blocking_cars))
    return len(list_of_blocking_cars)

def check_cars_blocking(map, v, blocking_cars):
    if v.orientation == 'V':
        if v.x - 1 >= 0:
            if map[v.y][v.x - 1] in CARS or map[v.y][v.x - 1] in TRUCKS:
                #print("Found car ", map[v.y][v.x - 1], " at: ", v.y, v.x-1, "left of ", v.id) #debug
                if map[v.y][v.x - 1] not in blocking_cars:
                    blocking_cars.append(map[v.y][v.x - 1])

        if v.end_x + 1 < 6:
            if map[v.end_y][v.end_x + 1] in CARS or map[v.end_y][v.end_x + 1] in TRUCKS:
                #print("Found car ", map[v.end_y][v.end_x + 1], " at: ", v.end_y, v.end_x+1, "right of ", v.id) #debug
                if map[v.end_y][v.end_x + 1] not in blocking_cars:
                    blocking_cars.append(map[v.end_y][v.end_x + 1])
    else: #v.orientation == 'H'
        index = 1
        while v.y - index >= 0:
            if map[v.y - index][v.x] in CARS or map[v.y - index][v.x] in TRUCKS:
                #print("Found car ", map[v.y - 1][v.x], " at: ", v.y-1, v.x, "above ", v.id) #debug
                if map[v.y - index][v.x] not in blocking_cars:
                    blocking_cars.append(map[v.y - index][v.x])
            index = index + 1

        index = 1
        while v.end_y + index < 6:
            if map[v.end_y + index][v.end_x] in CARS or map[v.end_y + index][v.end_x] in TRUCKS:
                #print("Found car ", map[v.end_y + 1][v.end_x], " at: ", v.end_y+1, v.end_x, "below ", v.id) #debug
                if map[v.end_y + index][v.end_x] not in blocking_cars:
                    blocking_cars.append(map[v.end_y + index][v.end_x])
            index = index + 1
    return blocking_cars


def get_priority(state):
    return state.priority

def RRHC(initial_state, visiable_states):
    visited_states = []
    if solved_state(initial_state):
        #print("RRHC local max: ", initial_state.path)
        return initial_state, visiable_states
    
    best_state = deepcopy(initial_state)
    #evaluate starting state
    best_state.priority = Heuristic_calculator(best_state, 4)
    best_state.priority = best_state.priority + 1
    visited_states.append(best_state)
    child_states = expand(best_state, [], visited_states)
    for child in child_states:
        child.priority = Heuristic_calculator(child, 4)
        if child not in visiable_states:
            visiable_states.append(child)
    child_states.sort(key=get_priority)
    

    while len(child_states) > 0:
        current_state = child_states.pop(0)
        if solved_state(current_state):
            #print_game_map(current_state.map)
            #print("RRHC local max: ", current_state.path)
            return current_state, visiable_states
        visited_states.append(current_state)

        if current_state.priority <= best_state.priority:
            child_states = expand(current_state, [], visited_states)

            for child in child_states:
                child.priority = Heuristic_calculator(child, 4)
                if child not in visiable_states:
                    visiable_states.append(child)
            child_states.sort(key=get_priority)
            best_state = deepcopy(current_state)
    #print("Did not find solved state using RR Hill Climbing")
    #print_game_map(best_state.map)
    #print(best_state.path)
    return best_state, visiable_states
    

class Rush_Hour():
    def __init__(self, game_data):
        #have all algorithms in this class
        self.visited = []
        self.goal_states = []
        self.BSF_Queue = []
        self.DSF_Stack = []
        self.A_Star_Queue = []

        game_map = create_game_map(game_data)
        self.list_of_vehicles = create_vehicles_list(game_map)
        self.start_state = board(game_map, self.list_of_vehicles, [])
        self.BSF_Queue.append(self.start_state)
        print_game_map(game_map)
        

    def Breadth_First_Search(self):
        start_time = time.time()
        Num_of_goal_states = 0
        list_of_goal_states = []
        while len(self.BSF_Queue) != 0:
            current_state = self.BSF_Queue.pop(0)  #could copy then remove if this doesnt work
            self.visited.append(current_state)
            if solved_state(current_state):
                if Num_of_goal_states == 0:
                    #print_game_map(current_state.map) #debug
                    print("BFS Sol: ", current_state.path)
                    end_time = time.time()
                    print("Time: ", end_time-start_time)
                    print("Number of Searched states: ", len(self.BSF_Queue) + len(self.visited))
                #print("found goal state: ") #debug
                Num_of_goal_states = Num_of_goal_states + 1
                self.goal_states.append(current_state)

                #add x path to exit
                #doesnt count if x is the one thats been moved
                #print map and path
                #maybe add to solutions list
                #break
            else:
                #append child nodes to BSF Queue instead of making it equal
                self.BSF_Queue = expand(current_state, self.BSF_Queue, self.visited)
                #if this ^ is passing by ref the i probs dont need to make it equal BSF queue
        #find_optimal(self.goal_states)
        #print optimal goal state by search solution list for smallest board path

    def iterative_deepening(self):
        start_time = time.time()
        global Number_of_searched_states
        Number_of_searched_states = 0
        depth = 1
        bottom_reached = False
        while not bottom_reached:
            goal_state, bottom_reached = iterative_deepening_DFS(self.start_state, 0, depth, [])
            if goal_state is not None:
                #print_game_map(goal_state.map)
                print("ID Sol: ", goal_state.path)
                end_time = time.time()
                print("Time: ", end_time-start_time)
                print("Number of Searched states: ", Number_of_searched_states)
                return None
            depth = depth + 1
            #print("increasing depth to " + str(depth))
        return None

    def A_Star(self, Number_of_Heuristics):
        start_time = time.time()
        self.visited = []
        self.A_Star_Queue = []
        child_states = []
        self.A_Star_Queue.append(self.start_state)

        while len(self.A_Star_Queue) > 0:
            current_state = self.A_Star_Queue.pop(0)
            if current_state in self.visited:
                """
                old_state = get_state(self.visited, current_state)
                if current_state.priority < old_state.priority:
                    self.visited.remove(old_state)
                else:
                    continue
                """
                continue #checking if the above code matters

            if solved_state(current_state):
                print("A* with heuristic ", Number_of_Heuristics, ": ")
                #print_game_map(current_state.map) #debug
                print("A* Sol: ", current_state.path)
                end_time = time.time()
                print("Time: ", end_time-start_time)
                print("Number of Searched states: ", len(self.visited) + 1)
                return True
                
            self.visited.append(current_state)

            #get children
            child_states = expand(current_state, [], self.visited)

            #get H/priotiy of children
            for child in child_states:
                child.priority = Heuristic_calculator(child, Number_of_Heuristics)
                self.A_Star_Queue.append(child) #add children to queue
                
            #sort queue
            self.A_Star_Queue.sort(key=get_priority)

    def Simple_Hill_Climbing(self):
        start_time = time.time()
        if solved_state(self.start_state):
                #print_game_map(start_state.map)
                print("SHC Sol: ", start_state.path)
                end_time = time.time()
                print("Time: ", end_time-start_time)
                return True
        
        visited_states = []
        self.start_state.priority = Heuristic_calculator(self.start_state, 4)
        self.start_state.priority = self.start_state.priority + 1
        visited_states.append(self.start_state)
        child_states = expand(deepcopy(self.start_state), [], visited_states)
        for child in child_states:
            child.priority = Heuristic_calculator(child, 4)
        best_state = deepcopy(self.start_state)

        while len(child_states) > 0:
            current_state = child_states.pop(0)
            if solved_state(current_state):
                #print_game_map(current_state.map)
                print("SHC Sol: ", current_state.path)
                end_time = time.time()
                print("Time: ", end_time-start_time)
                return True

            visited_states.append(current_state)

            if current_state.priority <= best_state.priority:
                child_states = expand(deepcopy(current_state), [], visited_states)

                for child in child_states:
                    child.priority = Heuristic_calculator(child, 4)
                best_state = deepcopy(current_state)
        print("Did not find solved state using Simple Hill Climbing")
        #print_game_map(best_state.map)
        print(best_state.path)
        return False
 
    def Steepest_Ascent_Hill_Climbing(self):
        start_time = time.time()
        child_states = []
        visited_states = []
        
        if solved_state(self.start_state):
                #print_game_map(start_state.map)
                print("SAHC Sol: ", start_state.path)
                end_time = time.time()
                print("Time: ", end_time-start_time)
                return True

        best_state = deepcopy(self.start_state)
        #evaluate starting state
        best_state.priority = Heuristic_calculator(best_state, 4)
        best_state.priority = best_state.priority + 1
        visited_states.append(best_state)
        child_states = expand(best_state, [], visited_states)
        for child in child_states:
            child.priority = Heuristic_calculator(child, 4)
        child_states.sort(key=get_priority)

        for counter in range(len(child_states)):
            current_state = child_states.pop(0)
            if solved_state(current_state):
                #print_game_map(current_state.map)
                print("SAHC Sol: ", current_state.path)
                end_time = time.time()
                print("Time: ", end_time-start_time)
                return True
            visited_states.append(current_state)

            if current_state.priority <= best_state.priority:
                child_states = expand(current_state, [], visited_states)

                for child in child_states:
                    child.priority = Heuristic_calculator(child, 4)
                child_states.sort(key=get_priority)
                best_state = deepcopy(current_state)
        
        print("Did not find solved state using Steepest Ascent Hill Climbing")
        #print_game_map(best_state.map)
        print(best_state.path)
        return False

    def Stochastic_Hill_Climbing(self):
        start_time = time.time()
        visited_states = []
        if solved_state(self.start_state):
                #print_game_map(start_state.map)
                print("SCHC Sol: ", start_state.path)
                end_time = time.time()
                print("Time: ", end_time-start_time)
                return True

        best_state = deepcopy(self.start_state)
        #evaluate starting state
        best_state.priority = Heuristic_calculator(best_state, 4)
        best_state.priority = best_state.priority + 1
        visited_states.append(best_state)
        child_states = expand(best_state, [], visited_states)
        for child in child_states:
            child.priority = Heuristic_calculator(child, 4)

        while len(child_states) > 0:
            random_index = random.randint(0, len(child_states)-1)
            random_state = child_states.pop(random_index)
            if solved_state(random_state):
                #print_game_map(random_state.map)
                print("SCHC Sol: ", random_state.path)
                end_time = time.time()
                print("Time: ", end_time-start_time)
                return True
            visited_states.append(random_state)

            if random_state.priority <= best_state.priority:
                child_states = expand(random_state, [], visited_states)

                for child in child_states:
                    child.priority = Heuristic_calculator(child, 4)
                best_state = deepcopy(random_state)
        
        print("Did not find solved state using Stochastic Hill Climbing")
        #print_game_map(best_state.map)
        print(best_state.path)
        return False

    def Random_Restart_Hill_Climbing(self):
        start_time = time.time()
        visiable_states = []
        self.start_state.priority = Heuristic_calculator(self.start_state, 4)
        self.start_state.priority = self.start_state.priority + 1
        current_optimal_state, visiable_states = RRHC(self.start_state, visiable_states)
        #find all states - then choose random
        for x in range(10):
            random_index = random.randint(0, len(visiable_states)-1)
            random_state = visiable_states[random_index]
            local_maximum, visiable_states = RRHC(random_state, visiable_states)

            if local_maximum.priority < current_optimal_state.priority:
                current_optimal_state = local_maximum
        print("RRHC Sol: ", current_optimal_state.path)
        end_time = time.time()
        print("Time: ", end_time-start_time)
        return True



#-------- Main -----------

RH_game_data = read_data()

""" 
#prints all games
game_num = 1
for data in RH_game_data:
    print("Problem ", game_num+1)
    print_game_map(create_game_map(data))
    game_num = game_num + 1 """

#game start
for game_num in range(len(RH_game_data)):
    print("Problem ", game_num + 1)
    if game_num == 1 or game_num == 13: #BFS getting stuck with these puzzles
        continue
    game = Rush_Hour(RH_game_data[game_num])
    print("\nBreadth First Search: ")
    game.Breadth_First_Search()
    print("\niterative_deepening: ")
    game.iterative_deepening()
    print("\nA Star: ")
    game.A_Star(1)
    game.A_Star(2)
    game.A_Star(3)
    print("\nHill Climbing: ")
    game.Simple_Hill_Climbing()
    game.Steepest_Ascent_Hill_Climbing()
    game.Stochastic_Hill_Climbing()
    game.Random_Restart_Hill_Climbing()






