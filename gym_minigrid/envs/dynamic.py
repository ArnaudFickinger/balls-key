from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np

TILE_PIXELS_AGENTS = 3

class DynamicEnv(MiniGridEnv):

    def __init__(
        self,
        size=6,
        width=None,
        height=None,
        init_pos=None,
        init_dir=1,
        n_obstacles=4,
        view_size=5,
        seed=0,
        max_steps=100,
        init_pos_switch=None,
        pixel_obs=True,
        partial_obs=True,
        simple_dynamics=False
    ):
        self.agent_start_pos = init_pos
        self.agent_start_dir = init_dir

        self.simple_dynamics = simple_dynamics

        self.n_obstacles = n_obstacles

        self.init_pos_switch = init_pos_switch

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            agent_view_size=view_size,
            # Set this to True for maximum speed
            see_through_walls=False
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.switch = Key()

        if self.init_pos_switch is not None:
            self.put_obj(self.switch, *self.init_pos_switch)

        else:
            self.place_obj(self.switch)

        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def _reward(self, i, rewards, reward=1):
        pass

    def _handle_switch(self, i, rewards, fwd_pos, fwd_cell):
        if self.switch_type_str=='light_switch':
            if self.moving_objects:
                self.moving_objects = False
            else:
                self.moving_objects = True
        elif self.switch_type_str=='hard_switch':
            self.moving_objects = False
        self.grid.set(*fwd_pos, self.agents[i])
        self.grid.set(*self.agents[i].pos, None)
        self.agents[i].pos = fwd_pos

    def _handle_pickup(self, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.type=='key':
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    self.moving_objects=False
            elif fwd_cell.type=='agent':
                if fwd_cell.carrying:
                    if self.carrying is None:
                        self.carrying = fwd_cell.carrying
                        fwd_cell.carrying = None

    def _handle_drop(self, fwd_pos, fwd_cell): #TODO Someone drop the key
        if self.carrying:
            if fwd_cell is None:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None
                self.moving_objects = True


    def seeObject(self, obs):
        for ob in obs.reshape((self.view_size**2,self.world.encode_dim)):
            if ob[0] == self.world.OBJECT_TO_IDX['ball']:
                return True
        return False

    def reset(self):
        obs = super().reset()
        obs = self.get_obs_render(obs, tile_size=TILE_PIXELS_AGENTS)
        self.moving_objects=True
        return obs

    def pos_2_idx(self, pos):
        return self.grid_size * pos[0] + pos[1]

    def step(self, action):
        self.step_count += 1

        done = False

        if self.moving_objects:
            for i_obst in range(len(self.obstacles)):
                old_pos = self.obstacles[i_obst].cur_pos
                if self.simple_dynamics:
                    a = [[1,0],[0,1], [-1,0], [0,-1]]
                    b = np.random.choice(4)
                    pert=a[b]
                else:
                    pert = [np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])]
                try:
                    self.place_obj(self.obstacles[i_obst], top=([old_pos[i] + pert[i] for i in range(2)]), size=(1, 1),
                                   max_tries=3)
                    self.grid.set(*old_pos, None)
                except:
                    pass

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos

        # Pick up an object
        elif action == self.actions.pickup:
            self._handle_pickup(fwd_pos, fwd_cell)

        # Drop an object
        elif action == self.actions.drop:
            self._handle_drop(fwd_pos, fwd_cell)

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        obs = self.get_obs_render(obs, tile_size=TILE_PIXELS_AGENTS)

        if self.moving_objects:
            reward = 0
        else:
            reward = 1

        return obs, reward, done, {}

class DynamicEnv6x6(DynamicEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

register(
    id='MiniGrid-Dynamic-6x6-v0',
    entry_point='gym_minigrid.envs:DynamicEnv6x6'
)
