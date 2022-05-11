import numpy as np
import cv2

# open map in grayscale
map = cv2.imread("map.png", 0)
# get the size of the map
HEIGHT, WIDTH = map.shape
# initial robot location
rx,ry,rtheta = (WIDTH/4, HEIGHT/4, 0)

# GLOBAL VARIABLES
# set up step
STEP=10
# convert angle from degree to radian
TURN=np.radians(25)
# standard deviation for step, turn, sensor, and position
SIGMA_STEP=0.5
SIGMA_TURN=np.radians(5)
SIGMA_SENSOR = 5
SIGMA_POS = 2
# number of particles generated
NUM_PARTICLES=3000

# read keyboard input
def get_input():
    fwd = 0
    turn = 0
    halt = False
    # display window until any keypress
    k = cv2.waitKey(0)
    # go forward
    if k == ord('w'):
        fwd = STEP
    # turn right
    elif k == ord('d'):
        turn = TURN
    # turn left
    elif k == ord('a'):
        turn = -TURN
    # go backward
    elif k == ord('s'):
        fwd = -STEP
    # press any other key to stop
    else:
        halt = True
    return fwd, turn, halt

# Move the robot with Gaussian noise
def move_robot(rx,ry,rtheta,fwd,turn):
    # add noise to the movement of robot
    fwd_noisy = np.random.normal(fwd,SIGMA_STEP,1)
    rx += fwd_noisy*np.cos(rtheta)
    ry += fwd_noisy*np.sin(rtheta)
    turn_noisy=np.random.normal(turn,SIGMA_TURN,1)
    rtheta += turn_noisy
    return rx,ry,rtheta

# initialize particle could
def init():
    # create particle array with first column for x position, second column for y position, third column for turn radians
    particles = np.random.rand(NUM_PARTICLES,3)
    particles *= np.array((WIDTH,HEIGHT,np.radians(360)))
    return particles

# move the particles
def move_particles(particles, fwd, turn):
    # add movement information of robot to particles
    particles[:,0] += fwd*np.cos(particles[:,2])
    particles[:,1] += fwd*np.sin(particles[:,2])
    particles[:,2] += turn
    # limit the particles within the map
    particles[:,0] = np.clip(particles[:,0], 0.0, WIDTH-1)
    particles[:,1] = np.clip(particles[:,1], 0.0, HEIGHT-1)
    return particles

# get value from robot's sensor
def sense(x,y,noisy=False):
    x = int(x)
    y = int(y)
    # normalized elevation value at y,x
    if noisy:
        return np.random.normal(map[y,x], SIGMA_SENSOR, 1)
    # get elevation value at y,x
    return map[y,x]

# compute particle weights
def compute_weights(particles, robot_sensor):
    # create 0 array to save weights for each particle
    errors = np.zeros(NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        # get elevation for each particle
        elevation = sense(particles[i,0], particles[i,1], noisy=False)
        errors[i] = abs(robot_sensor-elevation)
        # the larger the weights is, the closer the elevation values are to the robot position's value
        weights = np.max(errors)-errors
        # set weights at boundaries to 0
        weights[
            (particles[:,0]==0) | (particles[:,0]==WIDTH-1) | (particles[:,1]==0) | (particles[:,1]==HEIGHT-1)
        ]=0.0
        # increase sensitivity of weights
        weights=weights**3
    return weights

# resample the particles
def resample(particles, weights):
    # normalize the weights and use it as probabilities
    probabilities = weights/np.sum(weights)
    # resample the particles based on probabilities
    new_index = np.random.choice(
        NUM_PARTICLES,
        size=NUM_PARTICLES,
        p=probabilities
    )
    # get a new set of particles after resampling
    particles=particles[new_index,:]
    return particles

# add noise to the particles
def add_noise(particles):
    # get a new standard deviation for turn radian
    SIGMA_TURN = np.radians(10)
    # join the arrays to have the same shape of particles
    noise = np.concatenate(
        (
            # random samples from a normal distribution for x,y's position and turn radian
            np.random.normal(0, SIGMA_POS, (NUM_PARTICLES, 1)),
            np.random.normal(0, SIGMA_POS, (NUM_PARTICLES, 1)),
            np.random.normal(0, SIGMA_TURN, (NUM_PARTICLES, 1))
        ),
        axis=1
    )
    # add noise to the particles
    particles += noise
    return particles

# display robot, particles, and best guess
def display(map,rx,ry,particles):
    # convert map's color space
    lmap = cv2.cvtColor(map,cv2.COLOR_GRAY2BGR)
    # display particles
    if len(particles) > 0:
        for i in range(NUM_PARTICLES):
            cv2.circle(lmap, (int(particles[i,0]),int(particles[i,1])),1,(255,0,0),1)
    # display robot
    cv2.circle(lmap,(int(rx),int(ry)),5,(0,225,0),10)
    # display best guess by calucation the means for x and y position
    if len(particles) > 0:
        px = np.mean(particles[:,0])
        py = np.mean(particles[:,1])
        cv2.circle(lmap,(int(px),int(py)),5,(0,0,255),5)
    # display an image with circles
    cv2.imshow("robot localization",lmap)

# main routine
particles = init()
while True:
    display(map,rx,ry,particles)
    fwd,turn,halt=get_input()
    if halt:
        break
    rx,ry,rtheta=move_robot(rx,ry,rtheta,fwd,turn)
    particles=move_particles(particles,fwd,turn)
    if fwd != 0:
        robot_sensor = sense(rx,ry,noisy=True)
        weights=compute_weights(particles,robot_sensor)
        particles=resample(particles,weights)
        particles=add_noise(particles)
# close all the windows
cv2.destroyAllWindows()

