import numpy as np
import tqdm
import subprocess
from comet_ml import Experiment
from ns3gym import ns3env
from ns3gym.start_sim import find_waf_path

import matplotlib.pyplot as plt
from collections import deque
import time
import json
import os 
import glob

import csv

class Logger:
    def __init__(self, send_logs, tags, parameters, experiment=None):
        #self.stations = 5
        self.stations = 30
        self.send_logs = send_logs
        if self.send_logs:
            if experiment is None:
                try:
                    json_loc = glob.glob("./**/comet_token.json")[0]
                    with open(json_loc, "r") as f:
                        kwargs = json.load(f)
                except IndexError:
                    kwargs = {
                        "api_key": "vvn1oNhnykbKKH0KLPmu9TS5L",
                        "project_name": "rl-in-descentralized-wifi",
                        "workspace": "sheila-janota"
                    }

                self.experiment = Experiment(**kwargs)
            else:
                self.experiment = experiment
        self.sent_mb = 0
        self.speed_window = deque(maxlen=100)
        self.speed_window_lists = [deque(maxlen=100) for _ in range(self.stations)]
        self.speed_window0 = deque(maxlen=100)
        self.speed_window1 = deque(maxlen=100)
        
        
        
        self.step_time = None
        self.current_speed = 0
        self.current_speed0 = 0
        self.current_speed1 = 0
       
        if self.send_logs:
            if tags is not None:
                self.experiment.add_tags(tags)
            if parameters is not None:
                self.experiment.log_parameters(parameters)

    def begin_logging(self, episode_count, steps_per_ep, sigma, theta, step_time):
        self.step_time = step_time
        if self.send_logs:
            self.experiment.log_parameter("Episode count", episode_count)
            self.experiment.log_parameter("Steps per episode", steps_per_ep)
            self.experiment.log_parameter("theta", theta)
            self.experiment.log_parameter("sigma", sigma)

    def log_round(self, states, reward, cumulative_reward, info, loss, observations, step):
        self.experiment.log_histogram_3d(states, name="Observations", step=step)
        info = [[j for j in i.split("|")] for i in info]
        info = np.mean(np.array(info, dtype=np.float32), axis=0)
        #print("info-shape: ", info.shape)
        try:
            round_mb_list = []  # Initialize an empty list
            for i in range(self.stations):
                round_mb_list.append(info[i])
                #print("round_mb_list", round_mb_list)

            # round_mb = np.mean([float(i.split("|")[0]) for i in info])
            #round_mb = info[0]
            round_mb0 = info[0]
            #print("round_mb_0", round_mb0)
            round_mb1 = info[1]
            #print("round_mb_1", round_mb1)
            
        except Exception as e:
            print(info)
            print(reward)
            raise e
        #self.speed_window.append(round_mb)
        #self.speed_window.append(round_mb0+round_mb1)
        self.speed_window.append(sum(round_mb_list))
        for i in range(self.stations):
            self.speed_window_lists[i].append(round_mb_list[i])
        
        self.speed_window0.append(round_mb0)        
        self.speed_window1.append(round_mb1)      
        
        
        self.current_speed = np.mean(np.asarray(self.speed_window)/self.step_time)
        current_speed_list = []
        for i in range(self.stations):
            current_speed_list.append(np.mean(np.asarray(self.speed_window_lists[i]) / self.step_time))
            #print ("curr_speed_list:", current_speed_list) # speed window de cada station
            
        self.current_speed0 = np.mean(np.asarray(self.speed_window0)/self.step_time)
        self.current_speed1 = np.mean(np.asarray(self.speed_window1)/self.step_time)       
              
       
        
        #self.sent_mb += round_mb
        #self.sent_mb += round_mb0+round_mb1
        self.sent_mb += sum(round_mb_list)
        # CW = np.mean([float(i.split("|")[1]) for i in info])
        cw_list = []
        for i in range(self.stations, self.stations * 2):
            cw_list.append(info[i])

        #CW0 = info[2]
        #CW1 = info[3]
      
        # stations = np.mean([float(i.split("|")[2]) for i in info])
        #self.stations = info[nwifi * 2]#info[4]
        fairness = info[self.stations*2+1]#info[5]

        if self.send_logs:
            self.experiment.log_metric("Round reward", np.mean(reward), step=step)
            self.experiment.log_metric("Per-ep reward", np.mean(cumulative_reward), step=step)
            self.experiment.log_metric("Megabytes sent", self.sent_mb, step=step)
            self.experiment.log_metric("Round megabytes sent1", round_mb0, step=step)
            self.experiment.log_metric("Round megabytes sent2", round_mb1, step=step)
            
            
            self.experiment.log_metric("Station count", self.stations, step=step)
            self.experiment.log_metric("Current throughput", self.current_speed, step=step)
            #self.experiment.log_metric("Current throughput0", current_speed_list[0], step=step)
            #self.experiment.log_metric("Current throughput1", current_speed_list[1], step=step)
            #self.experiment.log_metric("Current throughput0", self.current_speed0, step=step)
            #self.experiment.log_metric("Current throughput1", self.current_speed1, step=step)        
            self.experiment.log_metric("Fairness index", fairness, step=step)            
            
            for i, obs in enumerate(observations):
                self.experiment.log_metric(f"Observation {i}", obs, step=step)
                
            all_loss_values = []

            for index, individual_loss in enumerate(loss):
                loss_values = list(individual_loss.get_loss().values())
                all_loss_values.extend(loss_values)
                    
            with open(f"lossFile_{self.stations}.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(all_loss_values) 
                
            with open(f"cwFile_{self.stations}.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(cw_list)
                
            with open(f"througputFile_{self.stations}.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(current_speed_list)
            """
            for index, individual_loss in enumerate(loss):
                fileName = f"lossFileSheila{index}.csv"
                with open(fileName, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(individual_loss.get_loss().values()) 
            """ 
            
    def log_episode(self, cumulative_reward, speed, step):
        if self.send_logs:
            #self.experiment.log_metric("Cumulative reward", cumulative_reward, step=step)
            self.experiment.log_metric("Speed", speed, step=step)

        self.sent_mb = 0
        self.last_speed = speed
        self.speed_window = deque(maxlen=100) 
        self.current_speed = 0

    def end(self):
        if self.send_logs:
            self.experiment.end()

class Teacher:
    """Class that handles training of RL model in ns-3 simulator

    Attributes:
        agent: agent which will be trained
        env (ns3-gym env): environment used for learning. NS3 program must be run before creating teacher
        num_agents (int): number of agents present at once
    """
    def __init__(self, env, num_agents, preprocessor):
        self.preprocess = preprocessor.preprocess
        self.env = env
        self.num_agents = num_agents
        self.CW = 16
        self.action = None              # For debug purposes        

    def dry_run(self, agents, steps_per_ep):
        obs_list = []
        obs = self.env.reset()
        for list_a in obs[0]:
            obs = self.preprocess(np.reshape(list_a, (-1, len(self.env.envs), 1)))
            obs_list.append(obs)

        with tqdm.trange(steps_per_ep) as t:
            for step in t:                
                actions_list = np.empty(shape=(0,), dtype=np.int32) 

                for i, agent in enumerate(agents):
                    self.actions = agent.act()
                    actions_list = np.append(actions_list, self.actions)
                    
                next_obs, reward, done, info = self.env.step(actions_list)
                next_obs_list = []                                      
                for list_b in next_obs[0]:
                    # obs = self.preprocess(np.reshape(next_obs, (-1, len(self.env.envs), 1)))
                    next_obs = self.preprocess(np.reshape(list_b, (-1, len(self.env.envs), 1)))
                    next_obs_list.append(next_obs)
                obs_list = next_obs_list

                if(any(done)):
                    break

    def eval(self, agents, simTime, stepTime, history_length, tags=None, parameters=None, experiment=None):
        for agent in agents:
            agent.load()
        steps_per_ep = int(simTime/stepTime + history_length)

        logger = Logger(True, tags, parameters, experiment=experiment)        
        try:
           logger.begin_logging(1, steps_per_ep, agents[0].noise.sigma, agents[0].noise.theta, stepTime)
        except  AttributeError:
           logger.begin_logging(1, steps_per_ep, None, None, stepTime)
        
        add_noise = False

        obs_dim = 1
        time_offset = history_length//obs_dim*stepTime

        try:
            self.env.run()
        except AlreadyRunningException as e:
            pass

        cumulative_reward = 0
        reward = 0
        sent_mb = 0
        obs_list = []

        obs = self.env.reset()
        for list_a in obs[0]:
            obs = self.preprocess(np.reshape(list_a, (-1, len(self.env.envs), obs_dim)))
            obs_list.append(obs)

        with tqdm.trange(steps_per_ep) as t:
            for step in t:
                self.debug = obs_list
                                
                actions_list = np.empty(shape=(0,), dtype=np.int32) 

                for i, agent in enumerate(agents):
                    # self.actions = agent.act(np.array(logger.stations, dtype=np.float32), add_noise)
                    self.actions = agent.act(np.array(obs_list[i], dtype=np.float32), add_noise)                    
                    actions_list = np.append(actions_list, self.actions)  
                    
                next_obs, reward, done, info = self.env.step(actions_list)
                next_obs_list = []                                      
                for list_b in next_obs[0]:
                    next_obs = self.preprocess(np.reshape(list_b, (-1, len(self.env.envs), obs_dim)))
                    next_obs_list.append(next_obs)

                cumulative_reward += np.mean(reward)

                #for i, agent in enumerate(agents):
                if step>(history_length/obs_dim):
                     logger.log_round(obs_list[0], reward, cumulative_reward, info, agents, np.mean(obs_list[0], axis=0)[0], step) 
                t.set_postfix(mb_sent=f"{logger.sent_mb:.2f} Mb", curr_speed=f"{logger.current_speed:.2f} Mbps")

                obs_list = next_obs_list

                if(any(done)):
                    break

        self.env.close()
        self.env = EnvWrapper(self.env.no_threads, **self.env.params)

        print(f"Sent {logger.sent_mb:.2f} Mb/s.\tMean speed: {logger.sent_mb/(simTime):.2f} Mb/s\tEval finished\n")

        logger.log_episode(cumulative_reward, logger.sent_mb/(simTime), 0)

        logger.end()
        return logger


    def get_shape(lst):
        if isinstance(lst, list):
            # Recursive call for nested lists
            inner_shape = get_shape(lst[0])
            return [len(lst)] + inner_shape
        else:
            # Base case for non-nested list
            return []
    
    def train(self, agents, EPISODE_COUNT, simTime, stepTime, history_length, send_logs=True, experimental=True, tags=None, parameters=None, experiment=None):
        steps_per_ep = int(simTime/stepTime + history_length)

        logger = Logger(send_logs, tags, parameters, experiment=experiment)              
        try:
            logger.begin_logging(EPISODE_COUNT, steps_per_ep, agents[0].noise.sigma, agents[0].noise.theta, stepTime)
        except  AttributeError:
            logger.begin_logging(EPISODE_COUNT, steps_per_ep, None, None, stepTime)

        add_noise = True

        obs_dim = 1
        time_offset = history_length//obs_dim*stepTime

        for i in range(EPISODE_COUNT):
            print(i)
            try:
                self.env.run()
            except AlreadyRunningException as e:
                pass

            if i>=EPISODE_COUNT*4/5:
                add_noise = False
                print("Turning off noise")

            cumulative_reward = 0
            reward = 0
            sent_mb = 0
            obs_list = []         

            
            obs = self.env.reset()                                       
            
            for list_a in obs[0]:
                obs = self.preprocess(np.reshape(list_a, (-1, len(self.env.envs), obs_dim)))
                obs_list.append(obs)            
            
            self.last_actions = [None]*len(agents)          
                      
            with tqdm.trange(steps_per_ep) as t:
                for step in t:
                    self.debug = obs_list
                                   
                    actions_list = np.empty(shape=(0,), dtype=np.int32)               
                     
                    for j, agent in enumerate(agents):                        
                        self.actions = agent.act(np.array(obs_list[j], dtype=np.float32), add_noise)                        
                        actions_list = np.append(actions_list, self.actions)                                              
                                      
                    next_obs, reward, done, info = self.env.step(actions_list)
                    next_obs_list = []                                      
                    for list_b in next_obs[0]:
                        next_obs = self.preprocess(np.reshape(list_b, (-1, len(self.env.envs), obs_dim)))
                        next_obs_list.append(next_obs)                                       
                                          
                    for k, agent in enumerate(agents):              
                        if self.last_actions[k] is not None and step>(history_length/obs_dim) and i<EPISODE_COUNT-1:
                            action = [np.array([actions_list[k]])]
                            agent.step(obs_list[k], action, reward, next_obs_list[k], done, 2)                          

                    cumulative_reward += np.mean(reward) 

                    self.last_actions = actions_list                  
                    
                    #for l, agent in enumerate(agents):                    
                    if step>(history_length/obs_dim):
                         logger.log_round(obs_list[0], reward, cumulative_reward, info, agents, np.mean(obs_list[0], axis=0)[0], i*steps_per_ep+step)
                    t.set_postfix(mb_sent=f"{logger.sent_mb:.2f} Mb", curr_speed=f"{logger.current_speed:.2f} Mbps")

                    obs_list = next_obs_list

                    if(any(done)):
                        break

            self.env.close()
            if experimental:
                self.env = EnvWrapper(self.env.no_threads, **self.env.params)

            for agent in agents:
                agent.reset()            
            
            print(f"Sent {logger.sent_mb:.2f} Mb/s.\tMean speed: {logger.sent_mb/(simTime):.2f} Mb/s\tEpisode {i+1}/{EPISODE_COUNT} finished\n")

            logger.log_episode(cumulative_reward, logger.sent_mb/(simTime), i)

        logger.end()
        print("Training finished.")
        return logger

class AlreadyRunningException(Exception):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

class EnvWrapper:
    def __init__(self, no_threads, **params):
        self.params = params
        self.no_threads = no_threads
        self.ports = [13968+i+np.random.randint(40000) for i in range(no_threads)]
        self.commands = self._craft_commands(params)

        self.SCRIPT_RUNNING = False
        self.envs = []

        self.run()
        for port in self.ports:
            env = ns3env.Ns3Env(port=port, stepTime=params['envStepTime'], startSim=0, simSeed=0, simArgs=params, debug=False)
            self.envs.append(env)

        self.SCRIPT_RUNNING = True

    def run(self):
        if self.SCRIPT_RUNNING:
            raise AlreadyRunningException("Script is already running")

        for cmd, port in zip(self.commands, self.ports):
            subprocess.Popen(['bash', '-c', cmd])
        self.SCRIPT_RUNNING = True

    def _craft_commands(self, params):
        try:
            waf_pwd = find_waf_path("./")
        except FileNotFoundError:
            import sys
            sys.path.append("../../")
            waf_pwd = find_waf_path("../../")

        command = f'{waf_pwd} --run "RLinWiFi-Decentralized-v13-30-station'
        for key, val in params.items():
            command+=f" --{key}={val}"

        commands = []
        for p in self.ports:
            commands.append(command+f' --openGymPort={p}"')

        return commands

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())

        return np.array(obs)

    def step(self, actions):
        next_obs, reward, done, info = [], [], [], []

        for i, env in enumerate(self.envs):
            no, rew, dn, inf, _ = env.step(actions)
            next_obs.append(no)
            reward.append(rew)
            done.append(dn)
            info.append(inf)

        return np.array(next_obs), np.array(reward), np.array(done), np.array(info)

    @property
    def observation_space(self):
        dim = repr(self.envs[0].observation_space).replace('(', '').replace(',)', '').split(", ")[2]
        return (self.no_threads, int(dim))

    @property
    def action_space(self):
        dim = repr(self.envs[0].action_space).replace('(', '').replace(',)', '').split(", ")[2]
        return (self.no_threads, int(dim))

    def close(self):
        time.sleep(5)
        for env in self.envs:
            env.close()
        # subprocess.Popen(['bash', '-c', "killall linear-mesh"])

        self.SCRIPT_RUNNING = False

    def __getattr__(self, attr):
        for env in self.envs:
            env.attr()
