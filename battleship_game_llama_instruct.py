# Imports
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from statistics import mode, StatisticsError
from random import choice
import accelerate
from pickle import dump

# Define game classes
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


class AgentPrompter:
    def __init__(self, agent_personality, board_size):
        self.board_size = board_size
        self.ships = self.place_ships([2, 2, 3, 3, 5])
        self.agent_personality = agent_personality # Adjective!!!!
        self.opponent = None
        self.winner = False

        self.already_attacked = []
        self.strikes = []
        self.hits = []
        
        starting_board = self.draw_ships()
        
        self.game_messages = [
            {
                "role": "system",
                "content": "You are an agent playing battleships against the user.  Your initial board looks like this (Your undamaged ships appear as O):\n%s\nYou cannot where your opponent's ships are until you hit them, but you can target his ships using the coordinates on this blank board:\n%s\nWhen playing board games, you generally act very %s, which affects the moves you play." % (starting_board, self.draw_board(self.strikes, self.hits), self.agent_personality)
            }
        ]
        
        self.chat_messages = [
            {
                "role": "system",
                "content": "You are an agent playing battleships against an user.  Your initial board looks like this (Your undamaged ships appear as O):\n%s\nWhen playing board games, you generally very %s, affects the chat messages you send.  As the game progresses, I will provide status updates, such as whether you hit or missed most recently.  You should use these status updates and and the previous messages you have exchanged with the user to chat" % (starting_board, self.agent_personality)
            }
        ]

    def set_opponent(self, opponent):
        self.opponent = opponent

    def prompt_next_move(self):
        game_messages = self.game_messages + [{"role": "system", "content": "This board shows where you have attacked the user's ships (your hits appear as X and your misses appear as O):\n%s\nIt is generally a good idea to attack near your previous successful hits, as they show that there is a ship in that area.  However, you should not just attack every spot in order because you must use logic and reasoning to decide which area to attack.  Where would you like to attack? Choose a number between 0 and %d that IS NOT in this list: %s.  You should respond with a single sentence of your reasoning BEFORE giving the number of the area that you want to attack." % (self.draw_board(self.strikes, self.hits), self.board_size ** 2 - 1, " or ".join([str(i) for i in self.already_attacked]))}]

        # Get attack from player
        if self.agent_personality == "player":
            print(self.render_for_player(game_messages))
            inp = int(input(" > "))

        # Get attack from LLM
        else:
            sampling_params = SamplingParams(n=64, max_tokens=8192, stop_token_ids=terminators, temperature=0.5, top_p=0.8)
            prompt = tokenizer.apply_chat_template(
                game_messages, 
                tokenize=False, 
                add_generation_prompt=True
            ).replace("<|eot_id|>", "")
            temp_prompt = prompt # + "I will attack spot number"
            while True:
                print(temp_prompt)
                outputs = LLM.generate(temp_prompt, sampling_params)[0]
                outputs = [[
                    word.strip(" .,;!?").replace("#", "").replace(",", "").replace("", "") for word in output.text.strip().split(" ")
                ] for output in outputs.outputs]
                
                print(outputs)
                
                int_outputs = []
                for output in outputs:
                    try:
                        for word in output[::-1]:
                            if word.isnumeric():
                                int_outputs.append(int(word))
                                break
                    except ValueError:
                        pass
                try:
                    inp = mode(int_outputs)
                    if inp in self.already_attacked:
                        temp_prompt = prompt + "Since I have already attacked %s," % " and ".join([str(i) for i in self.already_attacked])
                    else:
                        break
                except StatisticsError:
                    pass

        # Process attack
        self.already_attacked.append(inp)
        self.strikes.append(inp)
        self.hits.append(self.opponent.did_strike_hit(inp)[0])
        if self.hits[-1]:
            self.game_messages += [
                {"role": "agent", "content": "I will attack spot number %d" % inp},
                {"role": "system", "content": "You hit the user's ship on spot number %d" % inp}
            ]
            self.chat_messages.append({"role": "system", "content": "Agent attacked %d and hit" % inp})
            self.opponent.chat_messages.append({"role": "system", "content": "User attacked %d and hit" % inp})
        else:
            self.game_messages += [
                {"role": "agent", "content": "I will attack spot number %d" % inp},
                {"role": "system", "content": "You missed your the user's ship"}
            ]
            self.chat_messages.append({"role": "system", "content": "Agent attacked %d and missed" % inp})
            self.opponent.chat_messages.append({"role": "system", "content": "User attacked %d and missed" % inp})

        print("PLAYER:", self.agent_personality, self.chat_messages[-1])

        # Prompt for chat message
        if all([ship.is_sunk() for ship in self.opponent.ships]):
            self.winner = True
        else:
            self.prompt_chat()
    
    def prompt_chat(self):
        # Get chat message from the player
        if self.agent_personality == "player":
            print(self.render_for_player(self.chat_messages))
            inp = input(" > ")

        # Get chat message from LLM
        else:
            sampling_params = SamplingParams(n=1, max_tokens=512, stop_token_ids=terminators, temperature=0.6, top_p=0.9)
            prompt = tokenizer.apply_chat_template(
                self.chat_messages, 
                tokenize=False, 
                add_generation_prompt=True
            ).replace("<|eot_id|>", "")
            output = LLM.generate([prompt], sampling_params)[0].outputs[0]
            inp = output.text.strip()

        # Add chat to chat messages
        self.chat_messages.append({"role": "agent", "content": inp})
        self.opponent.chat_messages.append({"role": "user", "content": inp})

        # Play opponent's turn
        self.opponent.prompt_next_move()

    def render_for_player(self, messages):
        messages = ["%s: %s" %(mess["role"], mess["content"]) for mess in messages]
        return "\n".join(messages)
    
    def place_ships(self, sizes):
        ships = []
        coord_pairs = []
        for size in sizes:
            ship = Ship(size, self.board_size)
            while any([pair in coord_pairs for pair in ship.coords_as_pairs()]):
                ship = Ship(size, self.board_size)
    
            coord_pairs += list(ship.coords_as_pairs())
            ships.append(ship)
        return ships

    def draw_ships(self):
        points = []
        hits = []
        for ship in self.ships:
            points += list(ship.coords_as_pairs())
            hits += ship.hits
        return self.draw_board(points, hits)
    
    def did_strike_hit(self, point):
        x, y = point // self.board_size, point % self.board_size

        for ship in self.ships:
            if (x, y) in list(ship.coords_as_pairs()):
                ship.mark_hit(x, y)
                return True, ship
        return False, None
        
    def draw_board(self, points, hits):
        blank_board = [["%3.f" % (i * self.board_size + j) for j in range(self.board_size)] for i in range(self.board_size)]
    
        for point, hit in zip(points, hits):
            if type(point) == int:
                x, y = point // self.board_size, point % self.board_size
            else:
                x,y = point
                
            if hit:
                blank_board[x][y] = " X "
            else:
                blank_board[x][y] = " O "
    
        str_board = "\n".join([" | ".join(row) for row in blank_board])
        return str_board


if __name__ == "__main__":
    # Open Llama 3 8B
    model_id = "casperhansen/llama-3-70b-instruct-awq" # meta-llama/Meta-Llama-3-8B-Instruct" # ""
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    LLM = LLM(model=model_id, tensor_parallel_size=2, max_model_len=3124, gpu_memory_utilization=0.9, swap_space=80) #, max_num_seqs=1)
    
    terminators = [
        tokenizer.eos_token_id,
        # tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    for i in range(5):
        try:
            player_one = AgentPrompter("timidly", 7)
            player_two = AgentPrompter("confidently", 7)
            player_one.set_opponent(player_two)
            player_two.set_opponent(player_one)
            player_one.prompt_next_move()
            dump((player_one, player_two), open("%d_2.pkl" % i + 1, "wb"))
        except Exception:
            print(Exception)

        try:
            player_one = AgentPrompter("stupidly", 7)
            player_two = AgentPrompter("smartly", 7)
            player_one.set_opponent(player_two)
            player_two.set_opponent(player_one)
            player_one.prompt_next_move()
            dump((player_one, player_two), open("%d_3.pkl" % i + 1, "wb"))
        except Exception:
            print(Exception)

        try:
            player_one = AgentPrompter("angrily", 7)
            player_two = AgentPrompter("calmly", 7)
            player_one.set_opponent(player_two)
            player_two.set_opponent(player_one)
            player_one.prompt_next_move()
            dump((player_one, player_two), open("%d_1.pkl" % i + 1, "wb"))
        except Exception:
            print(Exception)