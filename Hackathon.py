#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import cv2
import random
import time
import math
import matplotlib.pyplot as plt


# In[2]:


import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym import spaces


# # Helper Functions

# In[25]:


#functionss
def collision_landing_pad(landing_pad_position, score):
    terminate = True
    return terminate

def out_of_frame(rocket_position):
    if rocket_position[0]>=1920 or rocket_position[0]<0 or rocket_position[1]>=1080 or rocket_position[1]<0 :
        return 1
    else:
        return 0
    
def ode(t, y, *args):
    alpha = args[-5] # associated to left engine
    beta = args[-4] # associated to right engine
    T_l = args[-3] # thrust of left engine
    T_c = args[-2] # thrust of center engine, no gimbal capabilities
    T_r = args[-1] # thrust of right engine
    g = args[0]
    m = args[1]
    Theta = args[2]
    a = args[3]
    h = args[4]
    
    phi = y[2]
    
    vel_x = y[3]
    vel_y = y[4]
    vel_phi = y[5]
    
    pos_x_dot = vel_x
    pos_y_dot = vel_y
    phi_dot = vel_phi
    
    vel_x_dot = 1/m * (-np.sin(beta+phi)*T_r -np.sin(alpha+phi)*T_l - np.sin(phi)*T_c)
    vel_y_dot = -g + 1/m * (np.cos(beta+phi)*T_r + np.cos(alpha+phi)*T_l + np.cos(phi)*T_c)
    
    dr = (a - h*np.tan(beta))*np.cos(beta)
    dl = (a - h*np.tan(-alpha))*np.cos(-alpha)
    
    vel_phi_dot = 1/Theta * (T_r * dr - T_l * dl)
    
    return np.array([pos_x_dot, pos_y_dot, phi_dot, vel_x_dot, vel_y_dot, vel_phi_dot])


def ode_linear(t, y, *args):
    # linearisation around
        # alpha = 0
        # beta = 0
    
    # not "fully" linear since control input (T_r, T_c, T_l) is multiplied with TVC deflection angle (alpha, beta)
    
    alpha = args[-5] # deflection angle associated to left engine
    beta = args[-4] # deflection angle associated to right engine
    T_l = args[-3] # thrust of left engine
    T_c = args[-2] # thrust of center engine, no gimbal capabilities
    T_r = args[-1] # thrust of right engine
    g = args[0]
    m = args[1]
    Theta = args[2]
    a = args[3]
    h = args[4]
    
    phi = y[2]
    
    vel_x = y[3]
    vel_y = y[4]
    vel_phi = y[5]
    
    pos_x_dot = vel_x
    pos_y_dot = vel_y
    phi_dot = vel_phi
    
    vel_x_dot = 1/m * (-(beta+phi)*T_r -(alpha+phi)*T_l -phi*T_c)
    vel_y_dot = -g + 1/m * (T_c + T_r + T_l)
    
    dr = a - h*beta
    dl = a - h*(-alpha)
    
    vel_phi_dot = 1/Theta * (T_r * dr - T_l * dl)
    
    return np.array([pos_x_dot, pos_y_dot, phi_dot, vel_x_dot, vel_y_dot, vel_phi_dot])

def rk4_e(f, y, h, t, *args):
    # runge kutte 4th order explicit
    tk_05 = t + 0.5*h
    yk_025 = y + 0.5 * h * f(t, y, *args)
    yk_05 = y + 0.5 * h * f(tk_05, yk_025, *args)
    yk_075 = y + h * f(tk_05, yk_05, *args)
    
    return y + h/6 * (f(t, y, *args) + 2 * f(tk_05, yk_025, *args) + 2 * f(tk_05, yk_05, *args) + f(t+h, yk_075, *args))

def scaling(pos_x, pos_y, width, height, uncovered=0.1):
    # scale flight path to frame

    # uncovered is part of frame that is not used to visualize flight path
    used_width = int(width * (1 - 2*uncovered))
    used_height = int(height * (1 - 2*uncovered))
    
    max_pos_x = max(pos_x)
    min_pos_x = min(pos_x)
    max_pos_y = max(pos_y)
    min_pos_y = min(pos_y)
    
    dist_x = max_pos_x - min_pos_x
    dist_y = max_pos_y - min_pos_y
    
    if dist_x>=dist_y:
        max_dist = dist_x + 1
    else:
        max_dist = dist_y + 1
    
    scaled_x = used_width/(max_dist)*pos_x
    scaled_x = scaled_x - (min(scaled_x) - int(uncovered*width))
    
    scaled_y = used_height/(max_dist)*pos_y
    scaled_y = scaled_y - (min(scaled_y) - int(uncovered*width))
    scaled_y = height - scaled_y
                           
    return scaled_x.astype(np.int32), scaled_y.astype(np.int32)

def rotation(phi, points):    
    x_rotate = points[:,0] * np.cos(phi) - points[:,1] * np.sin(phi)
    y_rotate = points[:,0] * np.sin(phi) + points[:,1] * np.cos(phi)
    
    return np.array([x_rotate, y_rotate]).T.astype(np.int32)


# # Custom Environment

# In[26]:


g = 1.625                   # m/s^2
m = 1000                    # mass of the spacecraft in kg
Theta = 1000                # moment of inertia of the spacecraft
b = 2                       # distance of thrust vector from center of gravity --> check the handout for the sketch
a = 1                       # distance of thrust vector from center of gravity --> check the handout for the sketch

h = 1/30   # time step size
t0 = 0                  # initial time
tn = 10                 # end time
linear = False           # using linear (True) or nonlinear equations of motion (False)

x0 = 250                # initial x position
y0 = 350                # initial y position
phi0 = math.radians(-15) # initial rotation angle of spacecraft (mathematically positive defined)
velx0 = -25             # initial x direction velocity (u)
vely0 = -25             # initial y direction velocity (v)
velphi0 = 0             # initial z direction velocity (w)

alphaBetaRate = math.radians(22.5)
thrust_rate = 200

time = np.linspace(t0, tn, int((tn-t0)/h)+1)

states = np.zeros((6, len(time)))

# data coming from 2D_flight_dynamic_propagation.py
pos_x = states[0,:-1]#.astype(np.int32)
pos_y = states[1,:-1]#.astype(np.int32)

phi = -states[2,:-1] # negative sign, because rotation matrix is defined incorrelty with regard to mathematical positive rotation

vel_x = states[3,:-1]
vel_y = states[4,:-1]

alpha = 0
beta = 0

max_thrust = 1500

T_l = 0
T_r = 0
T_c = 1625 * 1.2


# In[68]:


class SpacecraftEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, gravity = -9.81,
        mass: float = 1000.0,
        moment_inertia: float = 1000.0,
        thrust_rate:float = 200,
        alphaBetaRate = 0,
        a= 1,
        b= 2,
        linear: bool = False,
        h = h):
        
        super(SpacecraftEnv, self).__init__()
        
        #rocket initial data
        self.gravity = gravity
        self.mass = mass
        self.moment_inertia = moment_inertia
        self.rocket_position = [250, 350] #vec2 position
        self.rocket_velocity = [-25, -25] #vec2 velocity
        self.rocket_accel = [0, 0]
        
        self.rocket_angle = 0 #float angle
        self.rocket_angular_vel = 0 #float angular velocity
        self.rocket_angular_accel = 0
        self.thrust_rate = thrust_rate
        
        #landing pad initial data
        self.pad_position = [0, 0] #vec2 position
        self.pad_width = 5 #float width
        self.pad_height = 2 #float height
        
        #rocket side engines
        self.alpha = 0
        self.beta = 0
        self.alphaBetaRate = alphaBetaRate
        self.thrust_dynamic = 200
        
        self.thrust_left = 0
        self.thrust_right = 0
        self.thrust_center = mass * gravity * 1.2
        
        self.a = a
        self.b = b
        self.h = h

        # position history
        self.pos_history = [] # tuple of x,y
        self.angle_history = [] # tuple of theta, alpha, beta
        self.velocity_history = [] # tuple of xdot, ydot
        self.thrust_history = [] # tuple of left, center, right engine
        self.angular_history = [] # thetadots

        
        self.linear = linear
        
        # thrust engines (3)
        # control alpha beta
        self.action_space = spaces.Box(-1, 1, (5,), dtype='float32')
        
#        low = np.array(
#            [
#                # these are bounds for position
#                # realistically the environment should have ended
#                # long before we reach more than 50% outside
#                -1.5,
#                -1.5,
#                # velocity bounds is 5x rated speed
#                -5.0,
#                -5.0,
#                -math.pi,
#                -5.0,
#                -2.0,
#                -2.0,
#            ]
#        ).astype(np.float32)
#        high = np.array(
#            [
#                # these are bounds for position
#                # realistically the environment should have ended
#                # long before we reach more than 50% outside
#                1.5,
#                1.5,
#                # velocity bounds is 5x rated speed
#                5.0,
#                5.0,
#                math.pi,
#                5.0,
#                2.0,
#                2.0,
#            ]
#        ).astype(np.float32)
        low = np.array(
            [
                -500,
                -500,
                # velocity bounds is 5x rated speed
                -1000,
                -1000,
                -math.pi,
                -60.0,
                math.radians(-60),
                math.radians(-60),
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                2880,
                1920,
                # velocity bounds is 5x rated speed
                1000,
                1000,
                math.pi,
                60.0,
                math.radians(60),
                math.radians(60),
            ]
        ).astype(np.float32) 
        self.observation_space = spaces.Box(low, high)
        self.reason = ""

    def step(self, action):
        #CALCULATION
        # update via newton and runga kutta
        args = [self.gravity, self.mass, self.moment_inertia, self.a, self.b, self.alpha, self.beta, self.thrust_left, self.thrust_center, self.thrust_right]
        coor_sets = np.array([self.rocket_position[0], self.rocket_position[1], self.rocket_angle, self.rocket_velocity[0], self.rocket_velocity[1], self.rocket_angular_vel])
        t_rk = 0 #not so important, only for runga kutta formalism

        if self.linear:
            coor_sets = rk4_e(ode_linear, coor_sets, self.h, t_rk, args)
        else:
            coor_sets = rk4_e(ode, coor_sets, self.h, t_rk, *args)

        self.rocket_position[0] = coor_sets[0]
        self.rocket_position[1] = coor_sets[1]
        self.pos_history.append((self.rocket_position[0],self.rocket_position[1]))
        self.rocket_angle = coor_sets[2]
        self.rocket_velocity[0] = coor_sets[3]
        self.rocket_velocity[1] = coor_sets[4]
        self.velocity_history.append((self.rocket_velocity[0], self.rocket_velocity[1]))
        self.rocket_angular_vel = coor_sets[5]
        self.angular_history.append(self.rocket_angular_vel)
        #print(self.rocket_position)
        #print(self.rocket_velocity)
        
        #RENDERING
        #cv2.imshow('Spacecraft Env', self.img)
        #cv2.waitKey(1)
        #self.img = np.zeros((500, 500, 3), dtype='uint8')
        #Display Rocket
        #Display Landing Pad
        
        #REWARD and TERMINATION
        reward = 0
        k_1 = 100
        k_2 = 100
        k_3 = 100
        k_4 = 0
        k_5 = 100000
        k_6 = 100000
        k_7 = 100000
        k_8 = 100000
        k_9 = 100000
        k_10 = 10000
        ### 1
        distance_to_landing_pad = np.linalg.norm(np.array(self.rocket_position) - np.array(self.pad_position))
        reward -= distance_to_landing_pad * k_1
        ### 2
        velocity = self.rocket_velocity[1] * self.rocket_velocity[1] + self.rocket_velocity[0]  * self.rocket_velocity[0]
        velocity = velocity**0.5
        reward -= velocity * k_2
        ### 3
        reward -= abs(self.rocket_angle) * k_3
        ### 4
        reward -= abs(self.rocket_angular_vel) * k_4

        self.thrust_center += self.thrust_rate * action[0] * self.h
        self.thrust_right += self.thrust_rate * action[1] * self.h
        self.thrust_left += self.thrust_rate * action[2] * self.h
        self.thrust_history.append((self.thrust_left, self.thrust_center, self.thrust_right))
        self.alpha += self.alphaBetaRate * action[3] * self.h
        self.beta += self.alphaBetaRate * action[4] * self.h
        self.angle_history.append((self.rocket_angle, self.alpha, self.beta))
        ### 5
        if self.thrust_left > max_thrust:
            self.thrust_left = max_thrust
            reward -= k_5
        if self.thrust_center > max_thrust:
            self.thrust_center = max_thrust
            reward -= k_6
        if self.thrust_right > max_thrust:
            self.thrust_right = max_thrust
            reward -= k_7
        if self.thrust_left < 0:
            self.thrust_left = 0
            reward -= k_5
        if self.thrust_center < 0:
            self.thrust_center = 0
            reward -= k_6
        if self.thrust_right < 0:
            self.thrust_right = 0
            reward -= k_7
        ### 6
        max_gimbal = math.radians(60)
        if self.alpha > max_gimbal:
            self.alpha = max_gimbal
            reward -= k_8
        if self.beta > max_gimbal:
            self.beta = max_gimbal
            reward -= k_9
        if self.alpha < -max_gimbal:
            self.alpha = -max_gimbal
            reward -= k_8
        if self.beta < -max_gimbal:
            self.beta = -max_gimbal
            reward -= k_9
        ### 7
        eps = 0.05
        if self.rocket_position[1] <= math.sqrt(a**2+b**2):
            if abs(self.rocket_position[0]) <= eps:
                reward += k_10
            self.terminated = True
        #### 2
#        epsilon = 0.005
#        y_com = self.b * np.cos(self.rocket_angle) + abs(self.a * np.sin(self.rocket_angle))
#        if (abs(y_com - self.rocket_position[1]) <= epsilon):
#            self.terminated = True
#            print("#2: " + str(self.rocket_position[0]) + " "+ str(self.rocket_position[1]))
#            self.reason = "Rocket has landed"
#            if (self.rocket_velocity[1] <= 1):
#                
#                reward += 10000 * (1.001 - self.rocket_velocity[1])    
#            else:
#                reward -= 10000
#
#        ### 3
#        if (self.rocket_angle < math.radians(-30) or self.rocket_angle > math.radians(30)):
#            reward -= (abs(self.rocket_angle) - math.radians(30))
#        ### 4
#        if (self.rocket_position[1] < self.pad_position[1]):
#            reward -= 25
#            
        ### 5
#        if (out_of_frame(self.rocket_position)):
#            print("#5: " + str(self.rocket_position[0]) + " " + str(self.rocket_position[1]))
#            self.reason = "out of frame"
#            reward -= 10000
#            self.terminated = True
        info = {}     
        #update observation
        rocket_position_x = self.rocket_position[0]
        rocket_position_y = self.rocket_position[1]

        rocket_velocity_x = self.rocket_velocity[0]
        rocket_velocity_y = self.rocket_velocity[1]

        rocket_angle = self.rocket_angle
        rocket_angular_vel = self.rocket_angular_vel
        
        alpha = self.alpha
        beta = self.beta

        observation = [rocket_position_x, rocket_position_y, rocket_velocity_x, rocket_velocity_y, rocket_angle, rocket_angular_vel, alpha, beta]
        return observation, reward, self.terminated, info

    def reset(self):
        self.img = np.zeros((500, 500, 3), dtype='uint8')

        #reset rocket data
        self.rocket_position = [250, 350] #vec2 position
        self.rocket_velocity = [-25, -25] #vec2 velocity
        self.rocket_accel = [0, 0]
        
        self.rocket_angle = phi0 #float angle
        self.rocket_angular_vel = 0 #float angular velocity
        self.rocket_angular_accel = 0
        self.is_touching_pad = False #bool is_touching_pad
        
        #general data
        self.terminated = False
        self.reason = ""
        info = {}
        self.reward = 0

        # position history
        self.pos_history = [] # tuple of x,y
        self.angle_history = [] # tuple of theta, alpha, beta
        self.velocity_history = [] # tuple of xdot, ydot
        self.thrust_history = [] # tuple of left, center, right engine
        self.angular_history = [] # thetadots


        rocket_position_x = self.rocket_position[0]
        rocket_position_y = self.rocket_position[1]
        rocket_velocity_x = self.rocket_velocity[0]
        rocket_velocity_y = self.rocket_velocity[1]
        rocket_accel_x = self.rocket_accel[0]
        rocket_accel_y = self.rocket_accel[1]

        rocket_angle = self.rocket_angle
        angular_vel = self.rocket_angular_vel
        angular_accel = self.rocket_angular_accel
                
        #thrust_left = self.thrust_left
        #thrust_right = self.thrust_right
        #thrust_center = self.thrust_center
        
        alpha = self.alpha
        beta = self.beta

        observation = [rocket_position_x, rocket_position_y, rocket_velocity_x, rocket_velocity_y, rocket_angle, angular_vel, alpha, beta]
        
        return observation

    def render(self, mode="human"):
        return

    def close(self):
        cv2.destroyAllWindows()


# # Model

# In[69]:


env = SpacecraftEnv(
        gravity = g,
        mass = m,
        alphaBetaRate = alphaBetaRate,
        a = a,
        b = b,
        linear = False,
        h = h)


# In[70]:


episodes = 5
for episode in range(1, episodes + 1):
    observation = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        random_action = env.action_space.sample()
        observation, reward, done, info = env.step(random_action)
        total_reward += reward
        
    print("episode {} with score: {}".format(episode, total_reward))
    
env.close()


# In[47]:


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(50000), progress_bar=True)
model.save("PPO-Spaceraft_1")


# # Visualization

# In[ ]:


FPS = 1/h      
safe_name= 'test.mp4'
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (255, 255, 255)
thickness = 1

n_timesteps = np.shape(env.pos_history)[0]

#resolution
image_height = 1080
image_width = 1920
px = np.array([i[0] for i in env.pos_history] + [env.pos_history[-1][0]]*int(FPS))
py = np.array([i[1] for i in env.pos_history] + [env.pos_history[-1][1]]*int(FPS))
scaled_x, scaled_y = scaling(px, py, image_width, image_height)

flight_path = np.array([scaled_x, scaled_y]).T
scaling = 10

lander_polygon = np.array([[-scaling, 2*scaling], [scaling, 2*scaling], [scaling, -2*scaling], [0, -3*scaling], [-scaling, -2*scaling]])
right_fin_polygon = np.array([[1.3*scaling, 2*scaling], [2*scaling, 3*scaling], [2*scaling, 1*scaling], [1.3*scaling, 0], [1.3*scaling, 2*scaling]])
left_fin_polygon = np.copy(right_fin_polygon)
left_fin_polygon[:,0] = left_fin_polygon[:,0] * -1


video = cv2.VideoWriter(safe_name,cv2.VideoWriter_fourcc('m','p','4','v'), FPS, (image_width,image_height))
env.angle_history += [env.angle_history[-1]]*int(FPS)
env.thrust_history += [env.thrust_history[-1]]*int(FPS)
env.pos_history += [env.pos_history[-1]]*int(FPS)
env.angular_history += [env.angular_history[-1]]*int(FPS)
env.velocity_history += [env.velocity_history[-1]]*int(FPS)
for i in range(len(env.pos_history)):
    # image = 255 * np.ones((image_height,image_width,3), np.uint8)
    # replace with cv2.imread(imageFile)
    image = cv2.imread('background.png')

    # image = cv2.polylines(image, [flight_path.reshape(-1,1,2)], False, (255,255,255), 1, cv2.LINE_AA)
    
    image = cv2.fillPoly(image, [(rotation(env.angle_history[i][0], lander_polygon)+flight_path[i,:]).reshape(-1,1,2)], (208,193,155))
    image = cv2.fillPoly(image, [(rotation(env.angle_history[i][0], right_fin_polygon)+flight_path[i,:]).reshape(-1,1,2)], (49,44,203))
    image = cv2.fillPoly(image, [(rotation(env.angle_history[i][0], left_fin_polygon)+flight_path[i,:]).reshape(-1,1,2)], (49,44,203))

    exhaust_polygon_left = np.array([[-9, 2*scaling], [-7, (2 + 3*env.thrust_history[i][0]/max_thrust)*scaling], [-5, 2*scaling]])
    exhaust_polygon_right = np.array([[9, 2*scaling], [7, (2 + 3*env.thrust_history[i][2]/max_thrust)*scaling], [5, 2*scaling]])
    exhaust_polygon_center = np.array([[2, 2*scaling], [0, (2 + 3*env.thrust_history[i][1]/max_thrust)*scaling], [-2, 2*scaling]])
    
    image = cv2.fillPoly(image, [(rotation(env.angle_history[i][0] + env.angle_history[i][1], exhaust_polygon_left)+flight_path[i,:]).reshape(-1,1,2)], (39,188,248))
    image = cv2.fillPoly(image, [(rotation(env.angle_history[i][0], exhaust_polygon_center)+flight_path[i,:]).reshape(-1,1,2)], (39,188,248))
    image = cv2.fillPoly(image, [(rotation(env.angle_history[i][0] + env.angle_history[i][2], exhaust_polygon_right)+flight_path[i,:]).reshape(-1,1,2)], (39,188,248))
    
    
    image = cv2.polylines(image, [np.array([flight_path[-1,:] + np.array([35, 3*scaling]), flight_path[-1,:] + np.array([-35, 3*scaling])]).reshape(-1,1,2)], isClosed = False, color=(0,0.0), thickness = 5)


    image = cv2.putText(image, 'Horizontal Position: %i [m]' % int(env.pos_history[i][0]), (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Vertical Position: %i [m]' % int(env.pos_history[i][1]), (50, 70), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Horizontal Velocity: %i [m/s]' % int(env.velocity_history[i][0]), (50, 90), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Vertical Velocity: %i [m/s]' % int(env.velocity_history[i][1]), (50, 110), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Left Engine Thrust: %i [%%]' % int(env.thrust_history[i][0]/max_thrust*100), (50, 130), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Center Engine Thrust: %i [%%]' % int(env.thrust_history[i][1]/max_thrust*100), (50, 150), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Right Engine Thrust: %i [%%]' % int(env.thrust_history[i][2]/max_thrust*100), (50, 170), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Left Engine Gimbal Angle: %i [Degrees]' % int(-env.angle_history[i][1]*180/np.pi), (50, 190), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Right Engine Gimbal Angle: %i [Degrees]' % int(-env.angle_history[i][2]*180/np.pi), (50, 210), font, fontScale, color, thickness, cv2.LINE_AA)
    #if env.terminated:
    #reason = env.reason if env.reason != "" else "Out of time"
    #image = cv2.putText(image, f"Simulation is terminated with reason: {reason}", (50, 230), font, fontScale, color, thickness, cv2.LINE_AA)
    video.write(image)
video.release()


# # Plotting

# In[ ]:

plt.figure(figsize=(24,5))
plt.subplot(141)
plt.plot(time[:len(env.pos_history)], px, label='Position x')
plt.plot(time[:len(env.pos_history)], py, label='Position y')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend()
plt.grid()
angle = np.array([i[0] for i in env.angle_history])
plt.subplot(142)
plt.plot(time[:len(env.angle_history)], angle*180/np.pi, label='Rotation phi')
plt.xlabel('Time [s]')
plt.ylabel('Angle [°]')
plt.legend()
plt.grid()
velo_x = [i[0] for i in env.velocity_history]
velo_y = [i[1] for i in env.velocity_history]
plt.subplot(143)
plt.plot(time[:len(env.velocity_history)], velo_x, label='Velocity x')
plt.plot(time[:len(env.velocity_history)], velo_y, label='Velocity y')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.legend()
plt.grid()

plt.subplot(144)
plt.plot(time[:len(env.angular_history)], np.array(env.angular_history)*180/np.pi, label='Velocity phi')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [°/s]')
plt.legend()
plt.grid()

plt.show()


# In[ ]:




