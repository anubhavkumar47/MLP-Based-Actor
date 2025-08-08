from gym.spaces import Box
import numpy as np

class Environment:
    def __init__(self):
        super(Environment, self).__init__()
        # --- Environment Parameters ---
        self.size = np.array([300, 300, 300])
        self.num_IoTD = 5
        self.T = 150

        # --- UAV & IoTD Initial State ---
        self.start_position_A = np.array([0, 299, 0], dtype=float)
        self.start_position_B = np.array([299, 299, 200], dtype=float)
        self.iotd_position = np.array([
            [50, 50, 0], [75, 150, 0], [100, 100, 0], [100, 250, 0], [150, 150, 0]
        ], dtype=float)
        self.e_position = np.array([
            [250, 100, 0], [280, 200, 0], [175, 225, 0], [200, 200, 0], [100, 150, 0]
        ], dtype=float)

        # --- Physics & Communication Parameters ---
        self.R_min = 0.1
        self.P_tx_UAV = 0.5
        self.P_circuit_UAV = 0.05
        self.eta = 0.5
        self.beta_0 = 1e-3
        self.noise_power = 1e-13
        self.IoTD_idle_drain = 0.001
        self.IoTD_comm_drain = 0.1

        # --- Action & Observation Spaces ---
        action_dim = 6 + 1 + self.num_IoTD # UAV_A(3), UAV_B(3), rho(1), delta(5)
        low_action = np.array([-1.0] * 6 + [0.0] * (1 + self.num_IoTD))
        high_action = np.array([1.0] * action_dim)
        self.action_space = Box(low=low_action, high=high_action, dtype=np.float32)
        self.observation_space = Box(low= np.array([-1.0] * 16 ),high=np.array([1.0] * 16 ))
        
        # State variables are initialized in reset()
        self.reset()

    def _get_flat_state(self):
        """ Returns the state as a single flat vector """
        return np.concatenate((
            self.current_position_A / self.size[0], # Normalize positions
            self.current_position_B / self.size[0],
            self.AoI / self.T, # Normalize AoI
            self.energy_levels
        ))

    def reset(self):
        self.current_position_A = self.start_position_A.copy()
        self.current_position_B = self.start_position_B.copy()
        self.AoI = np.zeros(self.num_IoTD)
        self.energy_levels = np.ones(self.num_IoTD)
        self.time = 0
        self.E = 0
        self.A = 0
        self.done = False
        return self._get_flat_state()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx_A, dy_A, dz_A = action[0:3]
        dx_B, dy_B, dz_B = action[3:6]
        rho = action[6]
        delta = action[7:]

        r_A, r_E, r_P = 0, 0, 0
        self.time += 1
        if self.time >= self.T:
            self.done = True
            return self._get_flat_state(), 0, self.done, {"energy":0,'aoi':0}

        # Movement and Propulsion Energy
        past_position_A = self.current_position_A.copy()
        past_position_B = self.current_position_B.copy()
        self.current_position_A += np.array([dx_A, dy_A, dz_A]) * 20
        self.current_position_B += np.array([dx_B, dy_B, dz_B]) * 20

        # Boundary checks
        if not np.all((self.current_position_A >= 0) & (self.current_position_A <= self.size)):
            self.current_position_A = np.clip(self.current_position_A, 0, self.size)
            r_P -= 10
        if not np.all((self.current_position_B >= 0) & (self.current_position_B <= self.size)):
            self.current_position_B = np.clip(self.current_position_B, 0, self.size)
            r_P -= 10

        propulsion_energy = self._calculate_propulsion_energy(past_position_A, self.current_position_A) + \
                            self._calculate_propulsion_energy(past_position_B, self.current_position_B)
        self.E += propulsion_energy

        # IoTD Idle Drain & AoI Update
        self.energy_levels = np.maximum(0, self.energy_levels - self.IoTD_idle_drain)
        self.AoI += 1

        # Communication Phase
        selected_iotd = np.argmax(delta)
        time_comm = rho
        time_energy_transfer = time_comm * 0.5
        time_data_collection = time_comm * 0.5

        # Energy Transfer
        dist_UAV_IoTD = np.linalg.norm(self.current_position_A - self.iotd_position[selected_iotd])
        channel_gain = self.beta_0 / (dist_UAV_IoTD**2 + 1e-9)
        energy_harvested = self.P_tx_UAV * channel_gain * time_energy_transfer * self.eta
        if self.energy_levels[selected_iotd] < 0.99:
            r_P += 5 * energy_harvested * 100
        self.energy_levels[selected_iotd] = min(1.0, self.energy_levels[selected_iotd] + energy_harvested)
        uav_tx_energy = self.P_tx_UAV * time_energy_transfer
        self.E += uav_tx_energy

        # Data Collection
        secure_rates = self._calculate_secure_rates()
        uav_rx_energy = 0
        if self.energy_levels[selected_iotd] > self.IoTD_comm_drain and secure_rates[selected_iotd] > self.R_min:
            r_P += 10
            self.A += self.AoI[selected_iotd]
            r_A = -0.1 * self.AoI[selected_iotd]
            self.AoI[selected_iotd] = 0
            self.energy_levels[selected_iotd] -= self.IoTD_comm_drain
            uav_rx_energy = self.P_circuit_UAV * time_data_collection
            self.E += uav_rx_energy
        elif rho > 0.1:
            r_P -= 5

        # Final Reward
        r_E = -0.01 * (propulsion_energy + uav_tx_energy + uav_rx_energy)
        reward = r_A + r_E + r_P
        return self._get_flat_state(), reward, self.done, {"energy":self.E,'aoi':self.A}

   

    def _calculate_propulsion_energy(self, pos_start, pos_end):
        distance = np.linalg.norm(pos_end - pos_start)
        return 0.05 * distance**2

    def _calculate_secure_rates(self):
        p_iotd_tx = 0.1
        p_jam = 0.1
        dist_IoTD_to_A = np.linalg.norm(self.iotd_position - self.current_position_A, axis=1)
        dist_B_to_A = np.linalg.norm(self.current_position_A - self.current_position_B)
        h_IoTD_A = self.beta_0 / (dist_IoTD_to_A**2 + 1e-9)
        h_B_A = self.beta_0 / (dist_B_to_A**2 + 1e-9)
        rate_at_uav = np.log2(1 + (p_iotd_tx * h_IoTD_A) / (p_jam * h_B_A + self.noise_power))
        max_rate_at_eavesdropper = np.zeros(self.num_IoTD)
        for i in range(self.num_IoTD):
            dist_IoTD_e = np.linalg.norm(self.e_position - self.iotd_position[i], axis=1)
            dist_B_e = np.linalg.norm(self.e_position - self.current_position_B, axis=1)
            h_IoTD_e = self.beta_0 / (dist_IoTD_e**2 + 1e-9)
            h_B_e = self.beta_0 / (dist_B_e**2 + 1e-9)
            rate_at_e = np.log2(1 + (p_iotd_tx * h_IoTD_e) / (p_jam * h_B_e + self.noise_power))
            max_rate_at_eavesdropper[i] = np.max(rate_at_e)
        return np.maximum(0, rate_at_uav - max_rate_at_eavesdropper)
