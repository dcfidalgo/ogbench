# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
# ]
# ///
import random
import numpy as np
import itertools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def plot_state_function(state_function: np.ndarray, adjacency: np.ndarray, n_tiles_per_state: int = 15,
                        ax: plt.Axes = None, cmap: str = "Blues"):
    n = n_tiles_per_state
    single_state_tiles = np.arange(n * n).reshape(n, n)
    i, j = state_function.shape

    a_shape = adjacency.shape
    assert len(a_shape) == 2
    assert a_shape[0] == a_shape[1]

    assert i*j == a_shape[0]

    total_mask = np.pad(np.zeros([n * i, n * j]) != 0, 1, constant_values=True)

    for state in range(i * j):
        mask = np.zeros([n, n]) != 0
        mask[0, -1] = True
        mask[0, 0] = True
        mask[-1, 0] = True
        mask[-1, -1] = True
        if not ((state % j != 0) and adjacency[state, state - 1]):
            mask = mask | (single_state_tiles % n == 0)

        if not ((state <= (i * j - j - 1)) and adjacency[state, state + j]):
            mask = mask | (single_state_tiles > (n * n - n - 1))

        if not ((state % j < (j - 1)) and adjacency[state, state + 1]):
            mask = mask | (single_state_tiles % n == (n - 1))

        if not ((state >= j) and adjacency[state, state - j]):
            mask = mask | (single_state_tiles < n)

        x = state // j
        y = state % j

        total_mask[1 + n * x:1 + n * x + n, 1 + n * y:1 + n * y + n] = mask

    func_enlarged = np.pad(np.repeat(np.repeat(state_function, n, axis=0), n, axis=1), 1, mode="edge")

    masked_func = np.ma.masked_where(total_mask, func_enlarged)

    pcm = ax.imshow(masked_func, cmap=cmap)

    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False)  # labels along the bottom edge are off

    ax.set_facecolor('grey')

    return pcm


def draw_maze(i, j, adjacency, ax):
    return plot_state_function(np.ones([i, j]), adjacency, ax=ax)


def draw_maze_collection(maze_collection):
    n_mazes = maze_collection["info"]["n_mazes"]
    lx = maze_collection["info"]["lx"]
    lz = maze_collection["info"]["lz"]
    pi = maze_collection["info"]["pi"]
    _, axs = plt.subplots(n_mazes, 1, figsize=(2*n_mazes, 2))
    axs[0].set_title(f"pi_1={pi}")
    for i, (ax, maze) in enumerate(zip(axs, maze_collection["mazes"])):
        draw_maze(lx, lz, maze, ax=ax)
    plt.show()


def n_edges_grid(lx, lz):
    return (lx - 1) * lz + (lz - 1) * lx


def max_pi(lx, lz):
    v = lx * lz
    e_grid = n_edges_grid(lx, lz)
    e_spanning_tree = v - 1
    return e_grid - e_spanning_tree


def cell_maze_from_adjacency(
        adjacency,
        n,
        random_goal=False,
        random_start=False
):

    max_cells = n * 2 + 1

    cell_array = np.ones(shape=(max_cells, max_cells), dtype="int32")

    # Dig the vertex cells
    for i in range(1, max_cells, 2):
        for j in range(1, max_cells, 2):
            cell_array[i, j] = 0

    if random_start:
        # Randomize a start, around the bottom-left
        s_x = np.random.randint(0, (max_cells - 1) // 2 - 1) * 2 + 1
        s_y = np.random.randint(0, (max_cells - 1) // 2 - 1) * 2 + 1
        start = (s_x, s_y)
    else:
        s_x = 1
        s_y = 1
        start = (s_x, s_y)

    if random_goal:
        # Randomize a goal (not quite efficiently)
        goal = (max_cells - 2, max_cells - 2)
        minimum_goal_dist = 0.45 * max_cells  # (at least that far)
        for e_x in range(1, max_cells, 2):
            for e_y in range(1, max_cells, 2):
                e_x = np.random.randint(0, (max_cells - 1) // 2 - 1) * 2 + 1
                e_y = np.random.randint(0, (max_cells - 1) // 2 - 1) * 2 + 1
                d_s_e = np.sqrt((e_x - s_x) ** 2 + (e_y - s_y) ** 2)
                if d_s_e > minimum_goal_dist:
                    goal = (e_x, e_y)
                    break
    else:
        goal = (max_cells-2, max_cells-2)

    for i, j in itertools.product(range(n*n), range(n*n)):
        if adjacency[i, j] and i < j:
            if j-i == 1:
                cell_array[(i % n) * 2 + 2, (i // n) * 2 + 1] = 0
            else:
                cell_array[(i % n) * 2 + 1, (i // n) * 2 + 2] = 0

    return cell_array


class WilsonGridGraphGeneratorWLoops:
    """
    Implementation by Natalia García-Colín.
    
    Wilson's Algorithm is an algorithm to generate a
    uniform spanning tree using a loop erased random walk. 
    In this modification specific for grid graphs I've added the 
    possibility of introducing loops.

    Wilson's Algorithm:
    1. Choose a random cell and add it to the visited list.
    2. Choose another random cell (Don’t add to visited list).
       This is the current cell.
    3. Choose a random cell that is adjacent to the current cell
       (Don’t add to visited list). This is your new current cell.
    4. Save the direction that you traveled on the previous cell.
    5. If the current cell is not in the visited cells list:
       a. Go to 3
    6. Else:
       a. Starting at the cell selected in step 2, follow the arrows
          and remove the edges that are crossed.
       b. Add all cells that are passed into the visited list
    7. If all cells have not been visited
       a. Go to 2
    8  Random walk is finished when all cells have been visited. 
       (A spanning tree is generated)
    8. (Adding loops) Once a random tree has been generated the algorithm adds 'pi'
    loops by adding 'pi' edges independently at random.
    """
    def __init__(self, height, width, pi=0):
        """WilsonGridGraphGeneratorWLoops(int,int) -> WilsonGridGraphGeneratorWLoops
        Creates a grid graph generator with specified width and height.
        width: width of generated grid graph
        height: height of generated grid graph
        pi: number of extra edges, defaults to 0 if not provided """

        # Make grid bigger to capture edges
        self.width = 2 * width - 1  # Make grid wider to capture edges
        self.height = 2 * height - 1  # Make grid higher to capture edges
        self.actual_width = width
        self.actual_height = height
        self.pi = pi
        self.cell_number = self.actual_width * self.actual_height

        # grid of cells initialized with zero values
        self.grid = [[0 for j in range(self.width)] for i in range(self.height)]

        # declare instance variable
        self.visited = []  # visited cells
        self.unvisited = []  # unvisited cells
        self.path = dict()  # random walk path. Dictionary of the form, cell: directionNumber
        self.walls = []  # positions representing absent edges
        self.edges =[]  # positions representing edges of the graph

        # valid directions in random walk
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # adjacency matrix of grid graph initialized with zero values
        self.adjacency = [[0 for j in range(self.cell_number)] for i in range(self.cell_number)]

        # indicates whether a grid graph is generated
        self.generated = False

    def __str__(self):
        """WilsonGridGraphGeneratorWLoops.__str__() -> str
        outputs a string version of the grid graph"""
        out = ""
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == 1 and (i % 2 == 0 and j % 2 == 0): #vertices
                    out += "*"
                else:
                    if self.grid[i][j] == 1 and (i % 2 == 0 and j % 2 == 1):  # vertical edges
                        out += "–"
                    else:
                        if self.grid[i][j] == 1 and (i % 2 == 1 and j % 2 == 0):  # horizontal edges
                            out += "|"
                        else:
                            out += " "
            out += "\n"
        return out

    def get_grid(self):
        """WilsonGridGraphGeneratorWLoops.get_grid() -> list
        returns the maze grid"""
        return self.grid

    def generate_grid(self):
        """WilsonGridGraphGeneratorWLoops.generate_grid() -> None
        Generates the spanning tree according to the Wilson Loop Erased Random
        Walk Algorithm and adds pi edges independently at random"""
        # reset the grid before generation
        self.initialize_grid()

        # choose the first cell to put in the visited list (Step 1)
        current = self.unvisited.pop(random.randint(0, len(self.unvisited) - 1))
        self.visited.append(current)
        self.mark(current)

        # loop until all cells have been visited
        while len(self.unvisited) > 0:
            # choose a random cell to start the walk (Step 2)
            first = self.unvisited[random.randint(0, len(self.unvisited) - 1)]
            current = first
            # loop until the random walk reaches a visited cell
            while True:
                # choose direction to walk (Step 3)
                dirNum = random.randint(0, 3)
                # check if direction is valid. If not, choose new direction
                while not self.is_valid_direction(current, dirNum):
                    dirNum = random.randint(0, 3)
                # save the cell and direction in the path
                self.path[current] = dirNum
                # get the next cell in that direction
                current = self.get_next_cell(current, dirNum, 2)
                if current in self.visited:  # visited cell is reached (Step 5)
                    break

            current = first  # go to start of path
            # loop until the end of path is reached
            while True:
                # add cell to visited, remove cell from unvisited
                # and mark visited cell and used edge onto
                self.visited.append(current)
                self.unvisited.remove(current)  # (Step 6.b)
                self.mark(current)

                # follow the direction to next cell (Step 6.a)
                dirNum = self.path[current]
                crossed = self.get_next_cell(current, dirNum, 1)  # edge crossed
                self.mark(crossed)  # mark crossed edge in grid matrix

                current = self.get_next_cell(current, dirNum, 2)
                if (current in self.visited):  # end of path is reached
                    self.path = dict()  # clear the path
                    break

        # add loops
        if self.pi != 0:
            for r in range(self.width):
                for c in range(self.height):
                    if self.grid[r][c] == 0 and (bool(r % 2 == 0) != bool(c % 2 == 0)):
                        self.walls.append((r, c))

            for i in range(self.pi):
                current = self.walls.pop(random.randint(0, len(self.walls) - 1))
                self.mark(current)

        self.generated = True

    def generate_adjacency(self):
        """WilsonGridGraphGeneratorWLoops.generate_adjacency() -> list
                Generates and returns the adjacency matrix of the resulting graph"""
        self.generate_grid()
        # mark an adjacency for each vertex to its right and down cells if an edge is indeed present
        for r in range(self.actual_height):
            for c in range(self.actual_width):
                cell_index = r * self.actual_width + c             # cell = (r, c)
                down_cell_index = (r + 1) * self.actual_width + c  # down_cell = (r+1, c)
                right_cell_index = r * self.actual_width + c + 1   # right_cell = (r, c+1)
                edge_down = (2 * r + 1, 2 * c)      # down edge index in grid
                edge_right = (2 * r, 2 * c + 1)     # right edge index in grid

                # first do all cells except bottom row and right column
                if (r != self.actual_height-1) and (c != self.actual_width-1):
                    self.adjacency[cell_index][down_cell_index] = self.grid[edge_down[0]][edge_down[1]]
                    self.adjacency[cell_index][right_cell_index] = self.grid[edge_right[0]][edge_right[1]]
                elif (r == self.actual_height-1) and (c != self.actual_width-1):    # bottom row
                    self.adjacency[cell_index][right_cell_index] = self.grid[edge_right[0]][edge_right[1]]
                elif (r != self.actual_height-1) and (c == self.actual_width-1):    # right column
                    self.adjacency[cell_index][down_cell_index] = self.grid[edge_down[0]][edge_down[1]]
                else:
                    pass

        # The procedure above fills in adjacency matrix for the upper diagonal r<=c.
        # We now fill the lower diagonal.
        for r in range(self.cell_number):
            for c in range(self.cell_number):
                if r > c:
                    self.adjacency[r][c] = self.adjacency[c][r]

        return self.adjacency

    ## Private Methods ##
    ## Do Not Use Outside This Class ##

    def get_next_cell(self, cell, dirNum, fact):
        """WilsonMazeGenerator.get_next_cell(tuple,int,int) -> tuple
        Outputs the next cell when moved a distance fact in the the
        direction specified by dirNum from the initial cell.
        cell: tuple (y,x) representing position of initial cell
        dirNum: int with values 0,1,2,3
        fact: int distance to next cell"""
        dirTup = self.directions[dirNum]
        return cell[0] + fact * dirTup[0], cell[1] + fact * dirTup[1]

    def is_valid_direction(self, cell, dirNum):
        """WilsonMazeGenerator(tuple,int) -> boolean
        Checks if the adjacent cell in the direction specified by
        dirNum is within the grid
        cell: tuple (y,x) representing position of initial cell
        dirNum: int with values 0,1,2,3"""
        newCell = self.get_next_cell(cell, dirNum, 2)
        tooSmall = newCell[0] < 0 or newCell[1] < 0
        tooBig = newCell[0] > self.height or newCell[1] > self.width
        return not (tooSmall or tooBig)

    def initialize_grid(self):
        """WilsonMazeGenerator.initialize_grid() -> None
        Resets the maze grid to blank before generating a new maze."""
        for i in range(self.height):
            for j in range(self.width):
                self.grid[j][i] = 0

        # fill up unvisited cells. (These are represented by even
        # coordinate elements in the grid.)
        for r in range(self.height):
            for c in range(self.width):
                if r % 2 == 0 and c % 2 == 0:
                    self.unvisited.append((r, c))

        # fill up edge cells. (These are represented by elements
        # where one of the coordinates is odd.)
        for r in range(self.height):
            for c in range(self.width):
                if bool(r % 2 == 0) != bool(c % 2 == 0):
                    self.edges.append((r, c))

        self.visited = []
        self.path = dict()
        self.generated = False

    def mark(self, cell):
        """WilsonMazeGenerator.mark(tuple) -> None
        Sets the value of the grid at the location specified by cell
        to 1
        cell: tuple (y,x) location of where to mark"""
        self.grid[cell[0]][cell[1]] = 1