"""
A human-controlled policy that lets you play against an AI
"""
class Human:
    def __init__(self, board_size, win_size): pass

    def __call__(self, state):

        # Show human valid actions
        # valid_actions = self.domain.valid_actions_in(state)
        valid_actions = state.valid_actions()
        # print('actions', valid_actions)
    
        # Ask human for move (repeat if their input is invalid)
        while True:
            try:
                action = tuple(map(int, input("Enter action in format '<row>,<col>' (0-based index): ").split(",")))
                if action not in valid_actions: raise ValueError
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt("Game interrupted.")
            except:
                print("Invalid action, try again.")
    
        return action

