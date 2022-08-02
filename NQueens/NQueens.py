"""
N-Queens Problem:
-how can N queens be placed on an NxN chessboard so that they cannot attack each other
"""
import time

class state: #object for holding the chess board layout
    def __init__(self): #class constructor
        self.board = [[0] * N for i in range(N)] #chess board represented as a matrix

def print_solutions(sols): #function for displaying the chess board matrix 
    for x in sols: #cycles through and prints out each solutions board layout (matrix)
        for i in range(N): #loops through rows
            for j in range(N): #loops through columns
                print(x.board[i][j], end=" ")
            print(" ")
        print("\n")

def is_valid(board, row ,column): #function to check whether queen can be placed on the borad
    #check row, dont really need this but just double checking
    for i in range(N):
        if board[row][i] == 1:
            return False

    #check column
    for i in range(N):
        if board[i][column] == 1:
            return False

    #check upper left diagonal
    r = row
    c = column
    while r >= 0 and c >= 0:
        if board[r][c] == 1:
            return False
        r = r - 1
        c = c - 1

    #check upper right diagonal
    r = row
    c = column
    while r >= 0 and c < N:
        if board[r][c] == 1:
            return False
        r = r - 1
        c = c + 1
    
    #check lower left diagonal
    r = row
    c = column
    while r < N and c >= 0:
        if board[r][c] == 1:
            return False
        r = r + 1
        c = c - 1

    #check lower right diagonal
    r = row
    c = column
    while r < N and c < N:
        if board[r][c] == 1:
            return False
        r = r + 1
        c = c + 1

    return True #if no clashes were found then the queen can be placed in this spot

def BFS(valid_states, row): #reccursive function that checks where queens can go for each row of the chess board
    if row == N: #end of reccursion, return final listof solutions
        return valid_states

    elif row >= 1 and row < N:
        updated_states = [] #list of new/updated boards with a queen on the next row down
        for current_state in valid_states: #selects a board form valid states and checks if a queen can be placed somewhere in the next row
            for column in range(N): # goes through columns 0-7
                if(is_valid(current_state.board, row, column)): #checks if queen can be placed in this spot, then adds that board to the updated state list
                    new = current_state
                    new.board[row][column] = 1
                    updated_states.append(new)
        valid_states = BFS(updated_states, row+1) #calls function again the check all valid spots on the next row for updated chess boards

    elif row == 0: #add all first states to valid_states, placing a queen in each column on the first row
        for x in range(N):
            new = state() #creates new chess board state
            new.board[0][x] = 1 #puts in queen at first row column x
            valid_states.append(new)
        valid_states = BFS(valid_states,row+1)

    else:
        print("something wrong: row < 0 or > N")

    return valid_states


#------------------------main/driver code-------------------------------------------

user_input = input("Enter the size of the chess board NxN (1-20): ")
N = int(user_input)
solutions = [] #list for solutions including the board layout
row = 0
start_time = time.time() # runtime timer starts

if N > 0 and N < 21: # check to see if the board size is within the scope of the program
    solutions = BFS(solutions, int(row)) #runs Breadth first search to find solution

else: # if board size if less than 0 or greater than 20
    print("please enter a valid chess board size ")

if N > 0 and N < 7: #if the board size is between 1x1 and 6x6 then print the chess board
    print_solutions(solutions)

print("number of solutions for %d x %d board: %d " % (N,N,len(solutions)) ) #prints the number of solutions

end_time = time.time() #runtime timer stops
print("Run Time: ", end='') #prints run time
print(end_time-start_time) #actul runtime is put in a different print function because it gives a more exact value

#-----------------------pureado code/planning---------------------------------------
"""
class state:
    def__init__(self):
    self.board  = [[0 for w in range(N)] for h in range(N)]


def BFS(valid_states,row):#recursive bfs algo (does one level at a time)
    if row = 0: #add all first states to 
        for x in range(N):
            new = state
            state.board[0][x] = 1 #puts in all first row queens
            add new to valid_states
        valid_states = BFS(valid_states,row++)
        
    else if 1 <= row < N: #row in range(1,N-1)
        create new_valid_states
        for current_state in valid_states: #while valid_states is not empty- then recurs # might need to pop items somehow
            column = 0
            while column < N:
                if (is_valid(current_state.board, row, column)):
                    new = current_state
                    new.board[row][column] = 1
                    add new to new_valid_states
                column++
        #valid states should now be empty
        valid_states = BFS(new_valid_states, row++)
    
    if row = N:
        return valid_states

    else:
        print("something wrong")

    return valid_states

def is_valid(board, row ,column): #checks if queen can be placed on the borad
    #check row, dont really need this but just double checking
    for i in range(N-1):
        if board[row][i] == 1:
            return False

    #check column
    for i in range(N-1):
        if board[i][column] == 1:
            return False

    #check upper left diagonal
    r = row
    c = column
    while r >= 0 OR c >= 0:
        if board[r][c] == 1:
            return False
        r = r - 1
        c = c - 1

    #check upper right diagonal
    r = row
    c = column
    while r >= 0 OR c >= N -1:
        if board[r][c] == 1:
            return False
        r = r - 1
        c = c + 1
    
    #check lower left diagonal
    r = row
    c = column
    while r >= N - 1 OR c >= 0:
        if board[r][c] == 1:
            return False
        r = r + 1
        c = c - 1

    #check lower right diagonal
    r = row
    c = column
    while r >= N - 1 OR c >= N -1:
        if board[r][c] == 1:
            return False
        r = r + 1
        c = c + 1

    return True

"""

