import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from robot_vla_mujoco.mujoco_env.mujoco_parser import MuJoCoParserClass
from robot_vla_mujoco.mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from robot_vla_mujoco.mujoco_env.ik import solve_ik
from robot_vla_mujoco.mujoco_env.transforms import rpy2r, r2rpy, r2quat
import os
import copy
import glfw
import mujoco

# Default robot profiles
_ROBOT_PROFILES = {
    "omy": {
        "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        "gripper_type": "position",  # 4 independent position actuators
        "gripper_actuators": ["actuator_rh_r1", "actuator_rh_r2", "actuator_rh_l1", "actuator_rh_l2"],
        "gripper_joint": "rh_r1",  # joint to read gripper open/close state
        "ee_body": "tcp_link",
        "n_arm_joints": 6,
        "n_gripper_actuators": 4,
        "gripper_open_val": 1.0,
        "gripper_close_val": 0.0,
        "ik_target_pos": np.array([0.3, 0.0, 1.0]),
        "ik_target_rpy": np.deg2rad([90, -0., 90]),
        "plate_xyz": np.array([0.3, -0.25, 0.82]),
        "mug_red_range": {"x": [+0.32, +0.33], "y": [-0.00, +0.02], "z": [0.83, 0.83]},
        "mug_blue_range": {"x": [+0.29, +0.3], "y": [0.19, 0.21], "z": [0.83, 0.83]},
        "sim_settle_steps": 100,
        "viewer_distance": 2.0,
        "viewer_elevation": -30,
        "viewer_lookat": [0.3, 0.0, 0.5],
        "ik_max_tick": 200,
        "ik_stepsize": 1.0,
        "ik_eps": 1e-2,
        "ik_err_th": 1e-2,
    },
    "ur3e_ag95": {
        "joint_names": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        "gripper_type": "tendon",  # single tendon-driven actuator
        "gripper_actuators": ["fingers_actuator"],
        "gripper_joint": "fingers_actuator",  # actuator name for reading gripper state
        "ee_body": "tcp_link",
        "n_arm_joints": 6,
        "n_gripper_actuators": 1,
        "gripper_open_val": 0.0,
        "gripper_close_val": 0.9,
        "ik_target_pos": np.array([0.5, 0.0, 1.0]),
        "ik_target_rpy": np.deg2rad([90, -0., 90]),
        "plate_xyz": np.array([0.35, -0.30, 0.81]),
        "mug_red_range": {"x": [+0.30, +0.35], "y": [-0.05, +0.00], "z": [0.85, 0.85]},
        "mug_blue_range": {"x": [+0.30, +0.35], "y": [0.18, 0.23], "z": [0.85, 0.85]},
        "sim_settle_steps": 80,
        "viewer_distance": 3.0,
        "viewer_elevation": -25,
        "viewer_lookat": [0.5, 0.0, 0.5],
        "ik_max_tick": 300,
        "ik_stepsize": 2.0,
        "ik_eps": 1e-2,
        "ik_err_th": 5e-2,
        "oracle_config": {
            "ee_body": "oracle_grasp_body",
            "target_rpy_deg": (90.0, 0.0, 90.0),
            "pre_grasp_z_offset": 0.18,
            "grasp_z_offset": 0.02,
            "lift_z_offset": 0.20,
            "place_z_offset": 0.12,
            "ik_stepsize": 2.0,
            "ik_eps": 1e-2,
            "max_stage_steps": 50,
        },
    },
}


class SimpleEnv2:
    def __init__(self,
                 xml_path,
                 action_type='eef_pose',
                 state_type='joint_angle',
                 seed=None,
                 initialize_viewer=True,
                 robot_profile=None):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            seed: int, seed for random number generator
            robot_profile: str or dict, robot profile name ("omy", "ur3e_ag95") or dict override
        """
        # Resolve robot profile
        if robot_profile is None:
            robot_profile = "omy"
        if isinstance(robot_profile, str):
            if robot_profile not in _ROBOT_PROFILES:
                raise ValueError(f"Unknown robot profile '{robot_profile}'. Available: {list(_ROBOT_PROFILES.keys())}")
            self._rp = _ROBOT_PROFILES[robot_profile].copy()
        else:
            # Merge dict override with defaults
            defaults = _ROBOT_PROFILES.get(robot_profile, _ROBOT_PROFILES["omy"]).copy()
            defaults.update(robot_profile)
            self._rp = defaults

        self.joint_names = self._rp["joint_names"]
        self._gripper_type = self._rp["gripper_type"]
        self._gripper_actuators = self._rp["gripper_actuators"]
        self._gripper_joint = self._rp["gripper_joint"]
        self._ee_body = self._rp["ee_body"]
        self._n_arm_joints = self._rp["n_arm_joints"]
        self._n_gripper_actuators = self._rp["n_gripper_actuators"]
        self._gripper_open_val = self._rp["gripper_open_val"]
        self._gripper_close_val = self._rp["gripper_close_val"]

        print("[SimpleEnv2] Loading MuJoCo model...")
        # Load the xml file (verbose=False to suppress model info printout)
        self.env = MuJoCoParserClass(name='Tabletop', rel_xml_path=xml_path, verbose=False)
        self.action_type = action_type
        self.state_type = state_type

        # Finish the heavy reset path before opening the viewer window, otherwise
        # the OS marks the fresh window as unresponsive while initialization blocks.
        print("[SimpleEnv2] Preparing simulation state...")
        self.env.reset()
        self.reset(seed)
        if initialize_viewer:
            print("[SimpleEnv2] Opening viewer window...")
            self.init_viewer(reset_env=False)
        print("[SimpleEnv2] Environment ready.")

    def init_viewer(self, reset_env=True):
        '''
        Initialize the viewer
        '''
        if reset_env:
            self.env.reset()
        self.env.init_viewer(
            width             = 960,
            height            = 720,
            fontscale         = 150,
            distance          = self._rp["viewer_distance"],
            elevation         = self._rp["viewer_elevation"],
            lookat            = self._rp["viewer_lookat"],
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        print(f"[SimpleEnv2] Resetting task scene (seed={seed})...")
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        q_init = np.deg2rad([0,0,0,0,0,0])
        q_zero,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names,
            body_name_trgt     = self._ee_body,
            q_init       = q_init, # ik from zero pose
            p_trgt       = self._rp["ik_target_pos"],
            R_trgt       = rpy2r(self._rp["ik_target_rpy"]),
            max_ik_tick  = self._rp.get("ik_max_tick", 200),
            ik_stepsize  = self._rp.get("ik_stepsize", 1.0),
            ik_eps       = self._rp.get("ik_eps", 1e-2),
            ik_err_th    = self._rp.get("ik_err_th", 1e-2),
        )
        self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)

        # Initialize q for step_env() during settling
        self.last_q = copy.deepcopy(q_zero)
        if self._gripper_type == "tendon":
            self.q = np.concatenate([q_zero, np.array([self._gripper_open_val])])
        else:
            self.q = np.concatenate([q_zero, np.array([self._gripper_open_val] * self._n_gripper_actuators)])

        settle_short = max(20, self._rp["sim_settle_steps"] // 4)

        # Place freejoint task objects directly near the tabletop height. Dropping
        # the meshes from height can inject large contact impulses in the UR3e scene.
        self._set_free_body_pose('body_obj_plate_11', self._rp["plate_xyz"], np.eye(3, 3))

        # 2. Place red mug
        mr = self._rp["mug_red_range"]
        obj_xyzs = sample_xyzs(1, x_range=mr["x"], y_range=mr["y"], z_range=mr["z"],
                               min_dist=0.16, xy_margin=0.0)
        self._set_free_body_pose('body_obj_mug_5', obj_xyzs[0, :], np.eye(3, 3))

        # 3. Place blue mug
        mb = self._rp["mug_blue_range"]
        obj_xyzs = sample_xyzs(1, x_range=mb["x"], y_range=mb["y"], z_range=mb["z"],
                               min_dist=0.16, xy_margin=0.0)
        self._set_free_body_pose('body_obj_mug_6', obj_xyzs[0, :], np.eye(3, 3))

        for _ in range(settle_short):
            self.step_env()
        self._zero_free_body_velocity('body_obj_plate_11')
        self._zero_free_body_velocity('body_obj_mug_5')
        self._zero_free_body_velocity('body_obj_mug_6')
        mujoco.mj_forward(self.env.model, self.env.data)

        # Record initial TCP pose
        self.p0, self.R0 = self.env.get_pR_body(body_name=self._ee_body)
        mug_red_init_pose, mug_blue_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_red_init_pose, mug_blue_init_pose, plate_init_pose],dtype=np.float32)
        self.set_instruction()
        print(f"[SimpleEnv2] Reset complete. Task: {self.instruction}")
        self.gripper_state = False
        self.past_chars = []

    def _set_free_body_pose(self, body_name: str, p: np.ndarray, R: np.ndarray) -> None:
        jntadr = self.env.model.body(body_name).jntadr[0]
        qposadr = self.env.model.jnt_qposadr[jntadr]
        self.env.data.qpos[qposadr:qposadr + 3] = p
        self.env.data.qpos[qposadr + 3:qposadr + 7] = r2quat(R)
        self._zero_free_body_velocity(body_name)
        mujoco.mj_forward(self.env.model, self.env.data)

    def _zero_free_body_velocity(self, body_name: str) -> None:
        jntadr = self.env.model.body(body_name).jntadr[0]
        dofadr = self.env.model.jnt_dofadr[jntadr]
        self.env.data.qvel[dofadr:dofadr + 6] = 0.0
        self.env.data.qacc[dofadr:dofadr + 6] = 0.0

    def init_offscreen_renderer(self, width=640, height=480):
        """Initialize a MuJoCo Renderer for headless fixed-camera images."""
        self.offscreen_renderer = mujoco.Renderer(self.env.model, height=height, width=width)
        self.offscreen_width = width
        self.offscreen_height = height

    def close_offscreen_renderer(self):
        renderer = getattr(self, 'offscreen_renderer', None)
        if renderer is not None:
            renderer.close()
            self.offscreen_renderer = None

    def get_fixed_cam_rgb_offscreen(self, cam_name):
        renderer = getattr(self, 'offscreen_renderer', None)
        if renderer is None:
            raise RuntimeError("Offscreen renderer is not initialized. Call init_offscreen_renderer() first.")
        renderer.update_scene(self.env.data, camera=cam_name)
        return renderer.render().copy()

    def set_instruction(self, given = None):
        """
        Set the instruction for the task
        """
        if given is None:
            obj_candidates = ['red', 'blue']
            obj1 = random.choice(obj_candidates)
            self.instruction = f'Place the {obj1} mug on the plate.'
            if obj1 == 'red':
                self.obj_target = 'body_obj_mug_5'
            else:
                self.obj_target = 'body_obj_mug_6'
        else:
            self.instruction = given
            if 'red' in self.instruction:
                self.obj_target = 'body_obj_mug_5'
            elif 'blue' in self.instruction:
                self.obj_target = 'body_obj_mug_6'
            else:
                raise ValueError('Instruction does not contain a valid object color (red or blue).')

    def step(self, action):
        '''
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]

        '''
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = self._ee_body,
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 50,
                ik_stepsize        = 1.0,
                ik_eps             = 1e-2,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')

        # Build gripper command based on gripper type
        if self._gripper_type == "tendon":
            gripper_cmd = np.array([action[-1]])
        else:
            gripper_cmd = np.array([action[-1]] * self._n_gripper_actuators)
            if self._n_gripper_actuators == 4:
                gripper_cmd[[1, 3]] *= 0.8
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self, nstep=1):
        self.env.step(self.q, nstep=nstep)

    def grab_image(self, include_side=False):
        '''
        grab images from the environment
        returns:
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric
        '''
        if getattr(self, 'offscreen_renderer', None) is not None:
            self.rgb_agent = self.get_fixed_cam_rgb_offscreen(cam_name='agentview')
            self.rgb_ego = self.get_fixed_cam_rgb_offscreen(cam_name='egocentric')
        else:
            self.rgb_agent = self.env.get_fixed_cam_rgb(
                cam_name='agentview')
            self.rgb_ego = self.env.get_fixed_cam_rgb(
                cam_name='egocentric')
        self.rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        self.rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        if include_side:
            if getattr(self, 'offscreen_renderer', None) is not None:
                self.rgb_side = self.get_fixed_cam_rgb_offscreen(cam_name='sideview')
            else:
                self.rgb_side = self.env.get_fixed_cam_rgb(
                    cam_name='sideview')
            self.rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
        return self.rgb_agent, self.rgb_ego

    def grab_image_fast(self):
        """Grab raw camera images without overlay processing (for policy input)."""
        if getattr(self, 'offscreen_renderer', None) is not None:
            rgb_agent = self.get_fixed_cam_rgb_offscreen(cam_name='agentview')
            rgb_ego = self.get_fixed_cam_rgb_offscreen(cam_name='egocentric')
            return rgb_agent, rgb_ego
        rgb_agent = self.env.get_fixed_cam_rgb(cam_name='agentview')
        rgb_ego = self.env.get_fixed_cam_rgb(cam_name='egocentric')
        return rgb_agent, rgb_ego
        

    def render(self, teleop=False, idx = 0, fast=False, show_side_view=False):
        '''
        Render the environment
        '''
        if fast:
            self.env.render()
            return

        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name=self._ee_body)
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        self.env.plot_T(p = np.array([0.1,0.0,1.0]), label=f"Episode {idx}", plot_axis=False, plot_sphere=False)
        if hasattr(self, 'rgb_agent_view') and self.rgb_agent_view is not None:
            self.env.viewer_rgb_overlay(self.rgb_agent_view,loc='top right')
        if hasattr(self, 'rgb_egocentric_view') and self.rgb_egocentric_view is not None:
            self.env.viewer_rgb_overlay(self.rgb_egocentric_view,loc='bottom right')
        if (teleop or show_side_view) and hasattr(self, 'rgb_side_view') and self.rgb_side_view is not None:
            self.env.viewer_rgb_overlay(self.rgb_side_view, loc='top left')
        if teleop:
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
        if getattr(self, 'instruction', None) is not None:
            language_instructions = self.instruction
            self.env.viewer_text_overlay(text1='Language Instructions',text2=language_instructions)
        self.env.render()

    def _read_gripper_val(self) -> float:
        """Read current gripper state (0=open, 1=closed)."""
        if self._gripper_type == "tendon":
            try:
                idx = self.env.ctrl_names.index(self._gripper_joint)
                val = float(self.env.data.ctrl[idx])
            except (ValueError, IndexError):
                val = 0.0
            return 1.0 if val > 0.1 else 0.0
        else:
            gripper = self.env.get_qpos_joint(self._gripper_joint)
            return 1.0 if gripper[0] > 0.5 else 0.0

    def _read_gripper_raw(self) -> float:
        """Read raw gripper sensor/actuator value."""
        if self._gripper_type == "tendon":
            try:
                idx = self.env.ctrl_names.index(self._gripper_joint)
                return float(self.env.data.ctrl[idx])
            except (ValueError, IndexError):
                return 0.0
        else:
            return float(self.env.get_qpos_joint(self._gripper_joint)[0])

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper_cmd = self._read_gripper_val()
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)
    
    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation
        
        Keys:
            ---------     -----------------------
               w       ->        backward
            s  a  d        left   forward   right
            ---------      -----------------------
            In x, y plane

            ---------
            R: Moving Up
            F: Moving Down
            ---------
            In z axis

            ---------
            Q: Tilt left
            E: Tilt right
            UP: Look Upward
            Down: Look Donward
            Right: Turn right
            Left: Turn left
            ---------
            For rotation

            ---------
            z: reset
            SPACEBAR: gripper open/close
            ---------   


        '''
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0,-0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0,0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state =  not  self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        return action, False
    
    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper_cmd = self._read_gripper_val()
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def check_success(self):
        '''
        ['body_obj_mug_5', 'body_obj_plate_11']
        Check if the mug is placed on the plate
        + Gripper should be open and move upward above 0.9
        '''
        p_mug = self.env.get_p_body(self.obj_target)
        p_plate = self.env.get_p_body('body_obj_plate_11')
        if np.linalg.norm(p_mug[:2] - p_plate[:2]) < 0.1 and np.linalg.norm(p_mug[2] - p_plate[2]) < 0.6 and self._read_gripper_raw() < 0.1:
            p = self.env.get_p_body(self._ee_body)[2]
            if p > 0.9:
                return True
        return False
    
    def get_obj_pose(self):
        '''
        returns: 
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug
            p_plate: np.array, position of the plate
        '''
        p_mug_red = self.env.get_p_body('body_obj_mug_5')
        p_mug_blue = self.env.get_p_body('body_obj_mug_6')
        p_plate = self.env.get_p_body('body_obj_plate_11')

        return p_mug_red, p_mug_blue, p_plate
    
    def set_obj_pose(self, p_mug_red, p_mug_blue, p_plate):
        '''
        Set the object poses
        args:
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=p_mug_red)
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_mug_6',p=p_mug_blue)
        self.env.set_R_base_body(body_name='body_obj_mug_6',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name=self._ee_body)
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)
