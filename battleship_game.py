from vllm import LLM, SamplingParams
import pycatan
from random import choice
from statistics import mode

# TODO:
# Improve user playing experience

# Prompts
default_start_prompt = lambda opp_board, my_board, opp_last_action, my_last_action: """
You are playing battleships against an opponent.  The targeting board is shown below, where "X" represents a hit, "O" represents a miss, and areas that you can fire at are marked with a number.  Keep in mind that the other player's ships are not visible until you have hit them. :
%s

Your ships are shown below, where a destroyed part of a ship is marked with "X" and an untouched part of a ship is marked with "O".
%s
During your opponents turn, they %s.  During your last turn, you %s.
""" % (opp_board, my_board, opp_last_action, my_last_action)

personality_str = lambda personality: "  When playing board games, you %s, which affects the chats you send and moves you play." % personality[0]

def make_move(opp_board, my_board, opp_last_action, my_last_action, personality, strikes):
    str = default_start_prompt(opp_board, my_board, opp_last_action, my_last_action) 
    str += "Choose where to attack during your turn."
    if len(strikes) > 0:
        str += "  You have already attacked " + ", ".join(["%d" % strike for strike in strikes])
        str += ", so you cannot attack there again."
    str += personality_str(personality)
    str += "\nME: Because I have a %s personality, I will attack tile" % personality[1]
    return str

def comment(opp_board, my_board, opp_last_action, my_last_action, personality, other_comments, my_comments):
    str = default_start_prompt(opp_board, my_board, opp_last_action, my_last_action) + "There is a running chat log between you and your opponent." + personality_str(personality)
    str += "The chat log currently says"
    
    if len(other_comments) > len(my_comments): # Other user must have gone first
        for i in range(len(my_comments)):
            str += "\nOPPONENT: %s\nME: %s</s>" % (other_comments[i], my_comments[i])
        str += "\nOPPONENT: %s\nME:" % other_comments[-1]
    else:
        for i in range(len(my_comments)):
            str += "\nME: %s\nOPPONENT: %s</s>" % (my_comments[i], other_comments[i])
        str += "\nME:"
    return str


# Generate boards
class Ship:
    def __init__(self, length, board_size):
        self.left_right = choice([True, False])

        if self.left_right:
            x_start = choice(range(board_size-length))
            y_start = choice(range(board_size))
            self.x_coords = [x_start + i for i in range(length)]
            self.y_coords = [y_start for _ in range(length)]
        else:
            y_start = choice(range(board_size-length))
            x_start = choice(range(board_size))
            self.y_coords = [y_start + i for i in range(length)]
            self.x_coords = [x_start for _ in range(length)]
        self.hits = [False for _ in range(length)]

    def coords_as_pairs(self):
        return zip(self.x_coords, self.y_coords)

    def mark_hit(self, x, y):
        if self.left_right:
            self.hits[self.x_coords.index(x)] = True
        else:
            self.hits[self.y_coords.index(y)] = True

    def is_sunk(self):
        return all(self.hits)


def place_ships(sizes, board_size):
    ships = []
    coord_pairs = []
    for size in sizes:
        ship = Ship(size, board_size)
        while any([pair in coord_pairs for pair in ship.coords_as_pairs()]):
            ship = Ship(size, board_size)

        coord_pairs += list(ship.coords_as_pairs())
        ships.append(ship)
    return ships


def draw_board(points, hits, board_size):
    blank_board = [["%3.f" % (i * board_size + j) for j in range(board_size)] for i in range(board_size)]

    for point, hit in zip(points, hits):
        if type(point) == int:
            x, y = point // board_size, point % board_size
        else:
            x,y = point
            
        if hit:
            blank_board[x][y] = " X "
        else:
            blank_board[x][y] = " O "

    str_board = "\n".join([" | ".join(row) for row in blank_board])
    return str_board


def draw_ships(ships, board_size):
    points = []
    hits = []
    for ship in ships:
        points += list(ship.coords_as_pairs())
        hits += ship.hits
    return draw_board(points, hits, board_size)


def did_strike_hit(x, y, ships):
    for ship in ships:
        if (x, y) in list(ship.coords_as_pairs()):
            ship.mark_hit(x, y)
            return True, ship
    return False, None

# Load model
# LLM = LLM(model="meta-llama/Meta-Llama-3-8B", tensor_parallel_size=2)
LLM = LLM(model="lucyknada/microsoft_WizardLM-2-7B", tensor_parallel_size=2)

if __name__ == "__main__":
    # Setup game
    # Each personality should be a tuple like ("description starting with are", "one word description").  You can play by making it ("player", "player")
    PERSONALITIES = [("are confident and a strong player", "confident"), ("player", "player")] # ("are a worried, weak player", "worried")]
    BOARD_SIZE = 7
    SIZES = [2, 2, 3, 3, 5]
    
    both_sets_of_ships = [place_ships(SIZES, BOARD_SIZE), place_ships(SIZES, BOARD_SIZE)]
    last_actions = ["N/A", "N/A"]
    strikes = [[], []]
    hits = [[], []]
    comments = [[], []]
    
    # Game turn loop
    while all([len([ship for ship in ships if not ship.is_sunk()]) > 0 for ships in both_sets_of_ships]):
        for i in range(2):
            if PERSONALITIES[i][0] == "player":
                print(make_move(
                    draw_board(strikes[i], hits[i], BOARD_SIZE),
                    draw_ships(both_sets_of_ships[i], BOARD_SIZE),
                    last_actions[(i + 1) % 2], last_actions[i], PERSONALITIES[i], strikes[i]
                ))
                inp = int(input())
    
            else:
                sampling_params = SamplingParams(n=64, max_tokens=150)# temperature=0.8, top_p=0.90, n=16, max_tokens=50)
                outputs = LLM.generate([make_move(
                    draw_board(strikes[i], hits[i], BOARD_SIZE),
                    draw_ships(both_sets_of_ships[i], BOARD_SIZE),
                    last_actions[(i + 1) % 2], last_actions[i], PERSONALITIES[i], strikes[i]
                )], sampling_params)[0]
                outputs = [output.text.strip().split(" ")[0].strip() for output in outputs.outputs]
                
                int_outputs = []
                for out in outputs:
                    try:
                        out = out.replace("#", "").replace(",", "").replace("", "")
                        int_outputs.append(int(out))
                    except ValueError:
                        pass
    
                inp = mode(int_outputs)
    
            x, y = inp // BOARD_SIZE, inp % BOARD_SIZE
            strikes[i].append(inp)
            is_hit, ship = did_strike_hit(x, y, both_sets_of_ships[(i + 1) % 2])
            hits[i].append(is_hit)
    
            if is_hit:
                last_actions[i] = "struck %d, which was a hit" % inp
                if ship.is_sunk():
                    last_actions[i] += " and sunk their ship"
                print("SUNK", ship.is_sunk(), ship.hits)
            else:
                last_actions[i] = "struck %d, which was a miss" % inp

    
            if PERSONALITIES[i][0] == "player":
                print(comment(
                    draw_board(strikes[i], hits[i], BOARD_SIZE),
                    draw_ships(both_sets_of_ships[i], BOARD_SIZE),
                    last_actions[(i + 1) % 2],
                    last_actions[i],
                    PERSONALITIES[i],
                    comments[(i + 1) % 2],
                    comments[i]
                ))
                outputs = input()
    
            else:
                sampling_params = SamplingParams(n=1, max_tokens=150)# temperature=0.8, top_p=0.90, n=16, max_tokens=50)
                outputs = LLM.generate([comment(
                    draw_board(strikes[i], hits[i], BOARD_SIZE),
                    draw_ships(both_sets_of_ships[i], BOARD_SIZE),
                    last_actions[(i + 1) % 2],
                    last_actions[i],
                    PERSONALITIES[i],
                    comments[(i + 1) % 2],
                    comments[i]
                )], sampling_params)[0]
                outputs = [output.text.strip().split("</s>")[0].strip().split("\n")[0].strip().strip("</s>") for output in outputs.outputs][0]
    
            comments[i].append(outputs)
