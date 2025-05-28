import torch
import numpy as np
from gym.spaces import Box

class Environment:
    def __init__(self):
        super(Environment, self).__init__()
        # Define the size of the maze and the starting and target positions
        self.size = (300, 300, 300)
        self.num_IoTD = 5   #IoTD quantity
        self.start_position_A = np.array([0, 299, 150])     # Starting point of drone No. 1
        self.start_position_B = np.array([299,299, 200])   # Starting point of drone No. 2
        self.current_position_A = self.start_position_A
        self.current_position_B = self.start_position_B
        self.AoI = np.zeros(self.num_IoTD)    # AoI Counter
        self.R_min = 0.1
        self.time = 0# time slice
        self.T = 150   # Total time of a game
        self.E = 0  # Total energy consumption
        self.A = 0  # Total AoI
        self.avg_E = 0  # average energy consumption
        self.avg_A = 0  # Average AoI
        self.done = False


        # Define action space and observation space
        self.action_space = Box(low=np.array([-1] * 9), high=np.array([1] * 9))
        self.observation_space = Box(low=np.array([0] * 13), high=np.array([1] * 13))


        # Defining the maze layout
        self.iotd_position = np.array([
            [50, 50, 0], [75, 150, 0], [100, 100, 0], [100, 250, 0], [150, 150, 0]
        ])
        self.e_position = np.array([
            [250, 100, 0], [280, 200, 0], [175, 225, 0], [200, 200, 0], [100, 150, 0]
        ])

    def reset(self):
        # Reset the environment and put the agent in the starting position
        self.current_position_A = self.start_position_A
        self.current_position_B = self.start_position_B
        self.AoI = np.zeros(self.num_IoTD)
        self.time = 0  # time slice
        self.E = 0  # total energy consumption
        self.A = 0  # Total AoI
        self.avg_E = 0  # Average energy consumption
        self.avg_A = 0 # Average AoI
        UAV_A_position = np.array([self.start_position_A[0], self.start_position_A[1], self.start_position_A[2]])
        UAV_B_position = np.array([self.start_position_B[0], self.start_position_B[1], self.start_position_B[2]])
        energy_A = np.array([0])
        energy_B = np.array([0])
        self.done = False
        state = np.concatenate((UAV_A_position, UAV_B_position ,self.AoI , energy_A, energy_B))   # state
        return state

    def energy(self, pos_A, pos_B):
        energy = (79.85 * (1 + 3 * (np.linalg.norm(pos_A - pos_B) / 120) ** 2)
                    + 88.63 * np.sqrt(1 + 0.25 *
                                      (np.linalg.norm(pos_A - pos_B) / 4.03) ** 4)
                    - 0.5 * np.sqrt((np.linalg.norm(pos_A - pos_B) / 4.03) ** 2)
                    + 0.5 * 0.6 * 1.225 * 0.05 * 0.503 * np.linalg.norm(pos_A - pos_B) ** 3
                    )
        energy = np.array([energy])
        return  energy

    # action; horizontal movement, vertical movement, selected IOTD
    def step(self, action):

        # action shape: 2+2+5 2 controls movement, 5 controls selection        self.done = (self.time >= self.T)
        reward = 0
        # Resolve the action to a new location
        step_scale = 10
        action_A = np.array([action[0]*step_scale, action[1]*step_scale])
        action_B = np.array([action[2]*step_scale, action[3]*step_scale])
        choose_A = np.zeros((5, 1))
        for i in range(2, 7):
            choose_A[i - 2, 0] = action[i + 2]
        select_A = torch.argmax(torch.tensor(choose_A), dim=0)

        for i in range(0, 5):
            choose_A[i] = 0
            # choose_B[i] = 0
        choose_A[select_A] = 1

        new_A_x = int(self.current_position_A[0] + action_A[0])
        new_A_y = int(self.current_position_A[1] + action_A[1])
        new_A_z = int(self.current_position_A[2])

        new_B_x = int(self.current_position_B[0] + action_B[0])
        new_B_y = int(self.current_position_B[1] + action_B[1])
        new_B_z = int(self.current_position_B[2])

        if new_A_x < 0 or new_A_y > 299 or new_A_x > 299 or new_A_y < 0 or new_B_x < 0 or new_B_y > 299 or new_B_x > 299 or new_B_y < 0:
            # past_position_A = self.current_position_A  # Save the previous position for distance calculation
            # past_position_B = self.current_position_B
            if new_A_x < 0:
                new_A_x = 0
            if new_A_x >299:
                new_A_x = 299
            if new_A_y < 0:
                new_A_y = 0
            if new_A_y > 299:
                new_A_y = 299
            if new_B_x < 0:
                new_B_x = 0
            if new_B_x >299:
                new_B_x = 299
            if new_B_y < 0:
                new_B_y = 0
            if new_B_y > 299:
                new_B_y = 299
            self.current_position_A = np.array([new_A_x, new_A_y, new_A_z])
            self.current_position_B = np.array([new_B_x, new_B_y, new_B_z])
            for index, v in enumerate(self.AoI):
                self.AoI[index]+=1
            reward = -99
            self.time += 1
            energy_A = np.array([500])
            energy_B = np.array([500])
            state = np.concatenate((self.current_position_A, self.current_position_B ,self.AoI , energy_A, energy_B))
            return state, reward, self.done, 0,0

        # Update current location
        past_position_A = self.current_position_A   # Save the previous position for distance calculation
        past_position_B = self.current_position_B
        self.current_position_A = np.array([new_A_x, new_A_y, new_A_z])
        self.current_position_B = np.array([new_B_x, new_B_y, new_B_z])

        self.time += 1
        #Energy consumption per step
        energy_A = self.energy(self.current_position_A, past_position_A)
        energy_B = self.energy(self.current_position_B, past_position_B)

        # Calculate the distance from UAV to all IoTDs
        Distance_UAV_IoTD = np.zeros(5)
        for i in range(0, 5):
            dist = np.linalg.norm(self.current_position_A - self.iotd_position[i])
            Distance_UAV_IoTD[i] = dist
        #Calculate the distance from all IoTDs to all eavesdroppers
        Distance_IoTD_e = np.zeros((5, 5))
        for i in range(0, 5):
            for j in range(0, 5):
                dist = np.linalg.norm(self.iotd_position[i] - self.e_position[j])
                Distance_IoTD_e[i][j] = dist
        # Calculate the distance from UAV to Jammer
        Distance_UAV_Jammer = np.linalg.norm(self.current_position_A - self.current_position_B)
        # Calculate the distance from Jammer to all eavesdroppers
        Distance_Jammer_e = np.zeros(5)
        for i in range(0, 5):
            dist = np.linalg.norm(self.current_position_B - self.e_position[i])
            Distance_Jammer_e[i] = dist


        # safety assessment
        H_UAV_IoTD = np.zeros(5)
        H_IoTD_e = np.zeros((5, 5))
        H_UAV_Jammer = 0
        H_Jammer_e = np.zeros(5)

        for i in range(0, 5):
            H_UAV_IoTD[i] = (np.sqrt(0.001 / (Distance_UAV_IoTD[i] ** 2))
                             * np.sqrt(1 / 2)
                             * (np.e ** (-1j * 2 * np.pi * Distance_UAV_IoTD[i] / 0.12))
                             + np.sqrt(1 / 2) * np.random.normal(0, 1)
                             )
        for i in range(0, 5):
            for j in range(0, 5):
                H_IoTD_e[i][j] = (np.sqrt(0.001 / (Distance_IoTD_e[i][j] ** 2))
                                  *(np.sqrt(1 / 2)
                                  * (np.e ** (-1j * 2 * np.pi * Distance_IoTD_e[i][j] / 0.12)))
                                  + np.sqrt(1 / 2) * np.random.normal(0, 1)
                                  )
        H_UAV_Jammer = (np.sqrt(0.001 / (Distance_UAV_Jammer ** 2))
                                  *(np.sqrt(1 / 2)
                                  * (np.e ** (-1j * 2 * np.pi * Distance_UAV_Jammer / 0.12)))
                                  + np.sqrt(1 / 2) * np.random.normal(0, 1)
                        )
        for i in range(0, 5):
            H_Jammer_e[i] = (np.sqrt(0.001 / (Distance_Jammer_e[i] ** 2))
                             * np.sqrt(1 / 2)
                             * (np.e ** (-1j * 2 * np.pi * Distance_Jammer_e[i] / 0.12))
                             + np.sqrt(1 / 2) * np.random.normal(0, 1)
                             )

        R_D = np.zeros(5)
        R_E = np.zeros((5, 5))

        for i in range(0, 5):
            R_D[i] = np.log2(1 + 0.01 * (H_UAV_IoTD[i] ** 2) / (1e-13 + 0.001 * (H_UAV_Jammer ** 2)))

        for x in range(0, 5):
            for y in range(0, 5):
                R_E[x][y] = np.log2(1 + 0.01 * (H_IoTD_e[x][y] ** 2) / (1e-13 + 0.001 * H_Jammer_e[y]**2))

        def max_value(lst):
            # print(lst)
            max_value = -9999
            for item in lst:
                if item > max_value:
                    max_value = item
            return max_value

        R_sec = np.zeros(5)
        ttt = []
        for i in range(0, 5):      #The i-th IoTD
            for j in range(0,5):   #jth eavesdropper
                ttt.append(
                    R_D[i] - R_E[i][j]
            )
            R_sec[i] = (
                max(0, max_value(ttt))
            )

        self.A = 0
        for index, v in enumerate(self.AoI):
            if R_sec[index] > self.R_min and choose_A[index] == 1:
                self.A += self.AoI[index]
                self.AoI[index] = 1
                reward += 100
            else:
                self.AoI[index] += 1

        energy = energy_A + energy_B
        self.E += energy
        self.avg_E = self.E / self.time

        reward = reward - 0.1 * energy - 0.1*self.A
        reward = reward.squeeze(0)
        reward = reward.item()

        done = (self.time >= self.T)

        state = np.concatenate((self.current_position_A, self.current_position_B ,self.AoI , energy_A, energy_B))

        # Return new observation value, reward, whether it is finished, and additional information (optional)
        return state, reward, done ,energy,self.A