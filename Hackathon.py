#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[34]:


import numpy as np
import cv2
import random
import time
import math
import matplotlib.pyplot as plt


# In[35]:


import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym import spaces


# # Helper Functions

# In[36]:


#functionss
def collision_landing_pad(landing_pad_position, score):
    terminate = True
    return terminate

def out_of_frame(rocket_position):
    if rocket_position[0]>=1920 or rocket_position[0]<-500 or rocket_position[1]>=1080 or rocket_position[1]<-500 :
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
        max_dist = dist_x
    else:
        max_dist = dist_y
    
    scaled_x = used_width/(max_dist)*pos_x
    xoffset = min(scaled_x) - int(uncovered*width)
    scaled_x = scaled_x - xoffset
    
    scaled_y = used_height/(max_dist)*pos_y
    yoffset = min(scaled_y) - int(uncovered*width)
    scaled_y = scaled_y - yoffset
    scaled_y = height - scaled_y
                           
    return scaled_x.astype(np.int32), scaled_y.astype(np.int32), xoffset, yoffset

def rotation(phi, points):    
    x_rotate = points[:,0] * np.cos(phi) - points[:,1] * np.sin(phi)
    y_rotate = points[:,0] * np.sin(phi) + points[:,1] * np.cos(phi)
    
    return np.array([x_rotate, y_rotate]).T.astype(np.int32)


# # Custom Environment

# In[46]:


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
pos_x = []#states[0,:-1]#.astype(np.int32)
pos_y = []#states[1,:-1]#.astype(np.int32)

phi = []#-states[2,:-1] # negative sign, because rotation matrix is defined incorrelty with regard to mathematical positive rotation

vel_x = []#states[3,:-1]
vel_y = []#states[4,:-1]

alphalog = []
betalog = []

Tlog = []

alpha = 0
beta = 0

timer = 0

max_thrust = 1500

T_l = 0
T_r = 0
T_c = 1625 * 1.2

hello = []


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
        
        self.timer = 0
        self.prev_reward = 0
        self.linear = linear
        
        self.counter = 0
        
        # thrust engines (3)
        # control alpha beta
        self.action_space = spaces.Box(-1, 1, (5,), dtype='float32')
        
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
        
        pos_x.append(self.rocket_position[0])
        pos_y.append(self.rocket_position[1])
        
        self.rocket_angle = coor_sets[2]
        self.rocket_velocity[0] = coor_sets[3]
        self.rocket_velocity[1] = coor_sets[4]
        
        phi.append(-self.rocket_angle)
        vel_x.append(self.rocket_velocity[0])
        vel_y.append(self.rocket_velocity[1])

        self.rocket_angular_vel = coor_sets[5]
        
        #REWARD and TERMINATION
        reward = 0
        k_1 = 5
        k_2 = 100
        k_3 = 5
        k_4 = 0
        k_5 = 5
        k_6 = 5
        k_7 = 5
        k_8 = 5
        k_9 = 5
        k_10 = 10000
        k_11 = 5
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
        action1 = (action[1] - self.rocket_angle/(math.pi/2))/2
        action2 = (action[2] + self.rocket_angle/(math.pi/2))/2
        self.thrust_right += self.thrust_rate * action1 * self.h
        self.thrust_left += self.thrust_rate * action2 * self.h
        #self.thrust_right += self.thrust_rate * action[1] * self.h
        #self.thrust_left += self.thrust_rate * action[2] * self.h
        self.alpha += self.alphaBetaRate * action[3] * self.h
        self.beta += self.alphaBetaRate * action[4] * self.h
        alphalog.append(self.alpha)
        betalog.append(self.beta)
        Tlog.append((self.thrust_left, self.thrust_center, self.thrust_right))

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
            reward += k_10 * np.exp(-k_11 * abs(self.rocket_position[0]))
            self.terminated = True

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
        
        reward /= 20
        temp = reward
        reward -= self.prev_reward
        self.prev_reward = temp
        
        self.counter += 1
        hello.append(self.counter)
        
        self.timer += h
        #if (self.timer >= 10):
        #    self.terminated = True
            

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
        self.prev_reward = 0
        self.timer = 0

        rocket_position_x = self.rocket_position[0]
        rocket_position_y = self.rocket_position[1]
        rocket_velocity_x = self.rocket_velocity[0]
        rocket_velocity_y = self.rocket_velocity[1]
        rocket_accel_x = self.rocket_accel[0]
        rocket_accel_y = self.rocket_accel[1]

        rocket_angle = self.rocket_angle
        angular_vel = self.rocket_angular_vel
        angular_accel = self.rocket_angular_accel
                
        pos_x = []
        pos_y = []
        
        phi = [] # negative sign, because rotation matrix is defined incorrelty with regard to mathematical positive rotation
        
        vel_x = []
        vel_y = []

        alphalog = []
        betalog = []

        Tlog = []
        
        hello = []
        
        self.counter  = 0
                
        alpha = self.alpha
        beta = self.beta

        observation = [rocket_position_x, rocket_position_y, rocket_velocity_x, rocket_velocity_y, rocket_angle, angular_vel, alpha, beta]
        
        return observation

    def render(self, mode="human"):
        return

    def close(self):
        cv2.destroyAllWindows()


# # Model

# In[47]:


env = SpacecraftEnv(
        gravity = g,
        mass = m,
        alphaBetaRate = alphaBetaRate,
        a = a,
        b = b,
        linear = False,
        h = h)


# In[48]:


episodes = 1
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


# In[49]:


#log_path = "logging"
#model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=int(1), progress_bar=True)
#model.save("PPO-Spaceraft_1")


# # Visualization

# In[54]:


image_height = 1080
image_width = 1920

#for i in range(30):
#    pos_x[-i] = 0
#    pos_y[-i] = 0
pos_x = np.array(pos_x)
pos_y = np.array(pos_y)
phi = np.array(phi)
vel_x = np.array(vel_x)
vel_y = np.array(vel_y)

scaled_x, scaled_y, xoff, yoff = scaling(pos_x, pos_y, image_width, image_height, uncovered = 0.1)
n_timesteps = np.shape(pos_x)[0]

FPS = 1/h      
safe_name= 'test.mp4'
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (255, 255, 255)
thickness = 1




flight_path = np.array([scaled_x, scaled_y]).T

scaler = 10



lander_polygon = np.array([[-scaler, 2*scaler], [scaler, 2*scaler], [scaler, -2*scaler], [0, -3*scaler], [-scaler, -2*scaler]])
right_fin_polygon = np.array([[1.3*scaler, 2*scaler], [2*scaler, 3*scaler], [2*scaler, 1*scaler], [1.3*scaler, 0], [1.3*scaler, 2*scaler]])
left_fin_polygon = np.copy(right_fin_polygon)
left_fin_polygon[:,0] = left_fin_polygon[:,0] * -1
xoff = int(xoff)
yoff = int(yoff)
print(xoff,yoff)
video = cv2.VideoWriter(safe_name,cv2.VideoWriter_fourcc('m','p','4','v'), FPS, (image_width,image_height))
for i in range(n_timesteps-1):
    # image = 255 * np.ones((image_height,image_width,3), np.uint8)
    # replace with cv2.imread(imageFile)
    image = cv2.imread('background.png')
    
    
    # image = cv2.polylines(image, [flight_path.reshape(-1,1,2)], False, (255,255,255), 1, cv2.LINE_AA)
    
    image = cv2.fillPoly(image, [(rotation(phi[i], lander_polygon)+flight_path[i,:]).reshape(-1,1,2)], (208,193,155))
    image = cv2.fillPoly(image, [(rotation(phi[i], right_fin_polygon)+flight_path[i,:]).reshape(-1,1,2)], (49,44,203))
    image = cv2.fillPoly(image, [(rotation(phi[i], left_fin_polygon)+flight_path[i,:]).reshape(-1,1,2)], (49,44,203))

    image = cv2.fillPoly(image, [(rotation(phi[i], left_fin_polygon)+flight_path[i,:]).reshape(-1,1,2)], (49,44,203))

    exhaust_polygon_left = np.array([[-9, 2*scaler], [-7, (2 + 3*T_l/max_thrust)*scaler], [-5, 2*scaler]])
    exhaust_polygon_right = np.array([[9, 2*scaler], [7, (2 + 3*T_r/max_thrust)*scaler], [5, 2*scaler]])
    exhaust_polygon_center = np.array([[2, 2*scaler], [0, (2 + 3*T_c/max_thrust)*scaler], [-2, 2*scaler]])
    
    image = cv2.fillPoly(image, [(rotation(phi[i] + alpha, exhaust_polygon_left)+flight_path[i,:]).reshape(-1,1,2)], (39,188,248))
    image = cv2.fillPoly(image, [(rotation(phi[i], exhaust_polygon_center)+flight_path[i,:]).reshape(-1,1,2)], (39,188,248))
    image = cv2.fillPoly(image, [(rotation(phi[i] + beta, exhaust_polygon_right)+flight_path[i,:]).reshape(-1,1,2)], (39,188,248))
    
    image = cv2.polylines(image, [np.array([np.array([abs(xoff)-35, image_height - abs(yoff) + 3*scaler]), np.array([abs(xoff)+35, image_height - abs(yoff) + 3*scaler])]).reshape(-1,1,2)], isClosed = False, color=(255,0.0), thickness = 5)
    #image = cv2.polylines(image, [np.array([np.array([0-35, 888 + 3*scaler]), np.array([0+35, 888 + 3*scaler])]).reshape(-1,1,2)], isClosed = False, color=(255,0.0), thickness = 5)
    #image = cv2.polylines(image, [np.array([flight_path[-1,:] + np.array([35, 3*scaler]), flight_path[-1,:] + np.array([-35, 3*scaler])]).reshape(-1,1,2)], isClosed = False, color=(0,0.0), thickness = 5)

    image = cv2.putText(image, 'Horizontal Position: %i [m]' % int(pos_x[i]), (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Vertical Position: %i [m]' % int(pos_y[i]), (50, 70), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Horizontal Velocity: %i [m/s]' % int(vel_x[i]), (50, 90), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Vertical Velocity: %i [m/s]' % int(vel_y[i]), (50, 110), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Left Engine Thrust: %i [%%]' % int(Tlog[i][0]/max_thrust*100), (50, 130), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Center Engine Thrust: %i [%%]' % int(Tlog[i][1]/max_thrust*100), (50, 150), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Right Engine Thrust: %i [%%]' % int(Tlog[i][2]/max_thrust*100), (50, 170), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Left Engine Gimbal Angle: %i [Degrees]' % int(-alphalog[i]*180/np.pi), (50, 190), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Right Engine Gimbal Angle: %i [Degrees]' % int(-betalog[i]*180/np.pi), (50, 210), font, fontScale, color, thickness, cv2.LINE_AA)

    video.write(image)

video.release()


# # Plotting

# In[33]:


plt.figure(figsize=(24,5))
plt.subplot(141)
plt.plot(hello[:], pos_x, label='Position x')
plt.plot(hello[:], pos_y, label='Position y')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend()
plt.grid()

plt.subplot(142)
plt.plot(hello[:], phi, label='Rotation phi')
plt.xlabel('Time [s]')
plt.ylabel('Angle [°]')
plt.legend()
plt.grid()

plt.subplot(143)
plt.plot(hello[:], vel_x, label='Velocity x')
plt.plot(hello[:], vel_y, label='Velocity y')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.legend()
plt.grid()

#plt.subplot(144)
#plt.plot(time[:-1], states[5,:-1]*180/np.pi, label='Velocity phi')
#plt.xlabel('Time [s]')
#plt.ylabel('Velocity [°/s]')
#plt.legend()
#plt.grid()

plt.show()


# In[ ]:





# In[ ]:




