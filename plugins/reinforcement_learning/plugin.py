"""
Reinforcement Learning Plugin für WiFi-Scanning-Optimierung.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json
import time
from dataclasses import dataclass
from enum import Enum

from plugins import BasePlugin, PluginMetadata

try:
    import gym
    from gym import spaces
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    import joblib
except ImportError as e:
    logging.warning(f"PyTorch oder Gym nicht verfügbar: {e}")
    # Fallback-Implementierung ohne Deep RL
    torch = None
    gym = None

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Verfügbare Aktionen für den RL-Agent."""
    SCAN_CHANNEL_1 = 0
    SCAN_CHANNEL_6 = 1
    SCAN_CHANNEL_11 = 2
    SCAN_CHANNEL_36 = 3
    SCAN_CHANNEL_40 = 4
    SCAN_CHANNEL_44 = 5
    SCAN_CHANNEL_48 = 6
    PAUSE_SCANNING = 7
    INCREASE_SCAN_RATE = 8
    DECREASE_SCAN_RATE = 9

@dataclass
class WiFiEnvironmentState:
    """Zustand der WiFi-Umgebung."""
    current_channel: int
    channels_scanned: List[int]
    aps_found: int
    clients_found: int
    signal_strength: float
    scan_duration: float
    cpu_usage: float
    memory_usage: float
    last_scan_time: float
    total_events: int

class Plugin(BasePlugin):
    """Reinforcement Learning für WiFi-Scanning-Optimierung."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Reinforcement Learning",
            version="1.0.0",
            description="RL-basierte Optimierung für WiFi-Scanning",
            author="WLAN-Tool Team",
            dependencies=["gym", "torch", "numpy", "pandas", "joblib"]
        )
    
    def run(self, state: Dict, events: list, console, outdir: Path, **kwargs):
        """
        Führt RL-basierte Optimierung für WiFi-Scanning durch.
        """
        console.print("\n[bold cyan]Starte RL-basierte WiFi-Scanning-Optimierung...[/bold cyan]")
        
        try:
            # Umgebung erstellen
            env = self._WiFiScanningEnvironment(state, events)
            console.print(f"[green]WiFi-Umgebung erstellt mit {len(env.channels)} Kanälen[/green]")
            
            # Agent erstellen (versuche Deep RL, sonst Simple RL)
            try:
                if torch is not None:
                    agent = self._DeepRLAgent(
                        state_size=env.observation_space.shape[0],
                        action_size=env.action_space.n
                    )
                    agent_type = "Deep RL (PyTorch)"
                else:
                    raise ImportError("PyTorch nicht verfügbar")
            except ImportError:
                agent = self._SimpleRLAgent(
                    state_size=env.observation_space.shape[0],
                    action_size=env.action_space.n
                )
                agent_type = "Q-Learning"
            
            console.print(f"[green]RL-Agent erstellt: {agent_type}[/green]")
            
            # Training
            console.print("[cyan]Starte Training...[/cyan]")
            training_results = self._train_rl_agent(env, agent, episodes=50, console=console)
            
            # Evaluation
            console.print("[cyan]Führe Evaluation durch...[/cyan]")
            eval_rewards = []
            for _ in range(10):
                state_obs = env.reset()
                total_reward = 0
                while True:
                    action = agent.act(state_obs, training=False)
                    state_obs, reward, done, info = env.step(action)
                    total_reward += reward
                    if done:
                        break
                eval_rewards.append(total_reward)
            
            avg_eval_reward = np.mean(eval_rewards)
            console.print(f"[green]Durchschnittliche Belohnung (Evaluation): {avg_eval_reward:.2f}[/green]")
            
            # Optimierungsempfehlungen
            console.print("\n[bold]Optimierungsempfehlungen:[/bold]")
            
            # Beste Kanäle basierend auf Performance
            channel_performance = env.performance_metrics['channel_utilization']
            if channel_performance:
                best_channels = sorted(channel_performance.items(), key=lambda x: x[1], reverse=True)[:3]
                console.print(f"Beste Kanäle: {[f'Kanal {ch} ({count} Scans)' for ch, count in best_channels]}")
            
            # Scan-Effizienz
            efficiency = env.performance_metrics['scan_efficiency']
            console.print(f"Scan-Effizienz: {efficiency:.2f} (höher = besser)")
            
            # Ergebnisse speichern
            results = {
                'agent_type': agent_type,
                'training_results': training_results,
                'evaluation_rewards': eval_rewards,
                'performance_metrics': dict(env.performance_metrics),
                'recommendations': {
                    'best_channels': best_channels if channel_performance else [],
                    'scan_efficiency': efficiency,
                    'avg_eval_reward': avg_eval_reward
                }
            }
            
            # Agent speichern
            agent_file = outdir / "rl_agent.joblib"
            agent.save(agent_file)
            
            # Ergebnisse speichern
            results_file = outdir / "rl_optimization_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"[green]RL-Optimierung abgeschlossen. Ergebnisse gespeichert: {results_file}[/green]")
            console.print(f"[green]Trainierter Agent gespeichert: {agent_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]Fehler bei der RL-Optimierung: {e}[/red]")
            logger.error(f"Fehler bei der RL-Optimierung: {e}", exc_info=True)
    
    def _train_rl_agent(self, environment, agent, episodes: int = 100, console=None) -> Dict:
        """Trainiert den RL-Agent."""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            state = environment.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = agent.act(state, training=True)
                next_state, reward, done, info = environment.step(action)
                
                agent.learn(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            if console and episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                console.print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_epsilon': agent.epsilon
        }
    
    class _WiFiScanningEnvironment:
        """RL-Umgebung für WiFi-Scanning-Optimierung."""
        
        def __init__(self, state: Dict, events: list):
            self.state = state
            self.events = events
            self.current_step = 0
            self.max_steps = 1000
            
            # Kanal-Definitionen
            self.channels = [1, 6, 11, 36, 40, 44, 48]
            self.current_channel = 1
            self.scan_history = deque(maxlen=100)
            
            # Action und Observation Spaces
            self.action_space = spaces.Discrete(len(ActionType)) if gym else None
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(10,), dtype=np.float32
            ) if gym else None
            
            # Performance-Tracking
            self.performance_metrics = {
                'total_aps_found': 0,
                'total_clients_found': 0,
                'scan_efficiency': 0.0,
                'channel_utilization': defaultdict(int)
            }
        
        def reset(self) -> np.ndarray:
            """Setzt die Umgebung zurück."""
            self.current_step = 0
            self.current_channel = 1
            self.scan_history.clear()
            self.performance_metrics = {
                'total_aps_found': 0,
                'total_clients_found': 0,
                'scan_efficiency': 0.0,
                'channel_utilization': defaultdict(int)
            }
            return self._get_observation()
        
        def _get_observation(self) -> np.ndarray:
            """Erstellt den aktuellen Beobachtungsvektor."""
            # Normalisierte Features
            obs = np.zeros(10, dtype=np.float32)
            
            # Aktueller Kanal (normalisiert)
            obs[0] = self.current_channel / 48.0
            
            # Anzahl gefundener APs (normalisiert)
            obs[1] = min(self.performance_metrics['total_aps_found'] / 100.0, 1.0)
            
            # Anzahl gefundener Clients (normalisiert)
            obs[2] = min(self.performance_metrics['total_clients_found'] / 200.0, 1.0)
            
            # Scan-Effizienz
            obs[3] = self.performance_metrics['scan_efficiency']
            
            # Kanal-Auslastung (letzte 10 Scans)
            recent_channels = list(self.scan_history)[-10:]
            if recent_channels:
                channel_counts = defaultdict(int)
                for ch in recent_channels:
                    channel_counts[ch] += 1
                most_used = max(channel_counts.values()) if channel_counts else 0
                obs[4] = most_used / 10.0
            
            # Zeit seit letztem Scan
            obs[5] = min(self.current_step / 1000.0, 1.0)
            
            # CPU/Memory Usage (simuliert)
            obs[6] = np.random.random()  # Simuliert CPU-Usage
            obs[7] = np.random.random()  # Simuliert Memory-Usage
            
            # Event-Rate (simuliert)
            obs[8] = min(len(self.events) / 10000.0, 1.0) if self.events else 0.0
            
            # Scan-Dauer (normalisiert)
            obs[9] = min(self.current_step / 100.0, 1.0)
            
            return obs
        
        def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
            """Führt eine Aktion aus und gibt den neuen Zustand zurück."""
            self.current_step += 1
            
            # Aktion ausführen
            reward = self._execute_action(action)
            
            # Neuen Zustand beobachten
            observation = self._get_observation()
            
            # Episode beenden?
            done = self.current_step >= self.max_steps
            
            # Zusätzliche Informationen
            info = {
                'current_channel': self.current_channel,
                'aps_found': self.performance_metrics['total_aps_found'],
                'clients_found': self.performance_metrics['total_clients_found'],
                'scan_efficiency': self.performance_metrics['scan_efficiency']
            }
            
            return observation, reward, done, info
        
        def _execute_action(self, action: int) -> float:
            """Führt die gegebene Aktion aus und gibt die Belohnung zurück."""
            action_type = ActionType(action)
            reward = 0.0
            
            if action_type in [ActionType.SCAN_CHANNEL_1, ActionType.SCAN_CHANNEL_6, 
                              ActionType.SCAN_CHANNEL_11, ActionType.SCAN_CHANNEL_36,
                              ActionType.SCAN_CHANNEL_40, ActionType.SCAN_CHANNEL_44, 
                              ActionType.SCAN_CHANNEL_48]:
                # Kanal scannen
                channel_map = {
                    ActionType.SCAN_CHANNEL_1: 1,
                    ActionType.SCAN_CHANNEL_6: 6,
                    ActionType.SCAN_CHANNEL_11: 11,
                    ActionType.SCAN_CHANNEL_36: 36,
                    ActionType.SCAN_CHANNEL_40: 40,
                    ActionType.SCAN_CHANNEL_44: 44,
                    ActionType.SCAN_CHANNEL_48: 48
                }
                
                new_channel = channel_map[action_type]
                self.current_channel = new_channel
                self.scan_history.append(new_channel)
                
                # Simuliere Scan-Ergebnisse basierend auf echten Daten
                aps_found, clients_found = self._simulate_scan_results(new_channel)
                
                # Belohnung basierend auf gefundenen Geräten
                reward += aps_found * 0.1  # 0.1 Punkte pro AP
                reward += clients_found * 0.05  # 0.05 Punkte pro Client
                
                # Bonus für neue Kanäle
                if new_channel not in [ch for ch in self.scan_history[:-1]]:
                    reward += 0.2
                
                # Update Performance Metrics
                self.performance_metrics['total_aps_found'] += aps_found
                self.performance_metrics['total_clients_found'] += clients_found
                self.performance_metrics['channel_utilization'][new_channel] += 1
                
            elif action_type == ActionType.PAUSE_SCANNING:
                # Pausiere Scanning
                reward -= 0.1  # Kleine Strafe für Pausieren
                
            elif action_type == ActionType.INCREASE_SCAN_RATE:
                # Erhöhe Scan-Rate
                reward += 0.05  # Kleine Belohnung für Aktivität
                
            elif action_type == ActionType.DECREASE_SCAN_RATE:
                # Verringere Scan-Rate
                reward -= 0.02  # Kleine Strafe für Verlangsamung
            
            # Update Scan-Effizienz
            total_scans = len(self.scan_history)
            if total_scans > 0:
                unique_channels = len(set(self.scan_history))
                self.performance_metrics['scan_efficiency'] = unique_channels / total_scans
            
            return reward
        
        def _simulate_scan_results(self, channel: int) -> Tuple[int, int]:
            """Simuliert Scan-Ergebnisse basierend auf echten Daten."""
            # Basierend auf echten Events aus dem State
            aps_found = 0
            clients_found = 0
            
            # Zähle APs und Clients für diesen Kanal
            if hasattr(self.state, 'aps'):
                for ap in self.state.aps.values():
                    if hasattr(ap, 'channel') and ap.channel == channel:
                        aps_found += 1
            
            if hasattr(self.state, 'clients'):
                for client in self.state.clients.values():
                    if hasattr(client, 'last_channel') and client.last_channel == channel:
                        clients_found += 1
            
            # Füge etwas Zufälligkeit hinzu für Realismus
            aps_found += np.random.poisson(0.5)
            clients_found += np.random.poisson(1.0)
            
            return max(0, aps_found), max(0, clients_found)
    
    class _SimpleRLAgent:
        """Einfacher RL-Agent ohne Deep Learning (Q-Learning)."""
        
        def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1):
            self.state_size = state_size
            self.action_size = action_size
            self.learning_rate = learning_rate
            
            # Q-Table
            self.q_table = defaultdict(lambda: np.zeros(action_size))
            
            # Epsilon-Greedy Parameter
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            
            # Performance-Tracking
            self.episode_rewards = []
            self.episode_lengths = []
        
        def act(self, state: np.ndarray, training: bool = True) -> int:
            """Wählt eine Aktion basierend auf dem aktuellen Zustand."""
            state_key = tuple(state.round(2))  # Diskrete Zustände
            
            if training and np.random.random() <= self.epsilon:
                # Exploration
                return np.random.choice(self.action_size)
            else:
                # Exploitation
                return np.argmax(self.q_table[state_key])
        
        def learn(self, state: np.ndarray, action: int, reward: float, 
                  next_state: np.ndarray, done: bool):
            """Lernt aus der Erfahrung (Q-Learning)."""
            state_key = tuple(state.round(2))
            next_state_key = tuple(next_state.round(2))
            
            # Q-Learning Update
            current_q = self.q_table[state_key][action]
            next_q_max = np.max(self.q_table[next_state_key])
            
            target_q = reward + 0.95 * next_q_max * (not done)
            self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
            
            # Epsilon Decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        def save(self, filepath: Path):
            """Speichert das gelernte Modell."""
            model_data = {
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths
            }
            joblib.dump(model_data, filepath)
        
        def load(self, filepath: Path):
            """Lädt ein gespeichertes Modell."""
            if filepath.exists():
                model_data = joblib.load(filepath)
                self.q_table = defaultdict(lambda: np.zeros(self.action_size), model_data['q_table'])
                self.epsilon = model_data['epsilon']
                self.episode_rewards = model_data['episode_rewards']
                self.episode_lengths = model_data['episode_lengths']
    
    class _DeepRLAgent:
        """Deep RL-Agent mit PyTorch (falls verfügbar)."""
        
        def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
            if torch is None:
                raise ImportError("PyTorch ist nicht verfügbar")
            
            self.state_size = state_size
            self.action_size = action_size
            self.learning_rate = learning_rate
            
            # Neural Network
            self.network = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            )
            
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
            
            # Epsilon-Greedy Parameter
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
        
        def act(self, state: np.ndarray, training: bool = True) -> int:
            """Wählt eine Aktion basierend auf dem aktuellen Zustand."""
            if training and np.random.random() <= self.epsilon:
                return np.random.choice(self.action_size)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.network(state_tensor)
            return q_values.argmax().item()
        
        def learn(self, state: np.ndarray, action: int, reward: float, 
                  next_state: np.ndarray, done: bool):
            """Lernt aus der Erfahrung (Deep Q-Learning)."""
            # Vereinfachte Implementierung - in der Praxis würde man Experience Replay verwenden
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            current_q = self.network(state_tensor)[0][action]
            next_q_max = self.network(next_state_tensor).max()
            
            target_q = reward + 0.95 * next_q_max * (not done)
            loss = nn.MSELoss()(current_q, target_q.detach())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Epsilon Decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay