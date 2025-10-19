"""
Tests für das Reinforcement Learning Plugin.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import defaultdict

from plugins.reinforcement_learning.plugin import Plugin, ActionType, WiFiEnvironmentState
from wlan_tool.storage.state import WifiAnalysisState, ClientState


class TestReinforcementLearningPlugin:
    """Test-Klasse für das Reinforcement Learning Plugin."""
    
    @pytest.fixture
    def plugin(self):
        """Plugin-Instanz für Tests."""
        return Plugin()
    
    @pytest.fixture
    def mock_state(self):
        """Mock State mit Test-Clients."""
        state = WifiAnalysisState()
        
        # Erstelle Test-Clients (mindestens 10 für Training)
        device_types = ["smartphone", "laptop", "tablet", "iot_device", "router"]
        for i in range(15):
            mac = f"aa:bb:cc:dd:ee:{i:02x}"
            client = ClientState(mac=mac)
            client.device_type = device_types[i % len(device_types)]
            client.probe_requests = [f"SSID_{i}"]
            client.first_seen = 1000.0
            client.last_seen = 1100.0
            client.rssi_history = [-50, -55, -60]
            state.clients[mac] = client
        
        return state
    
    @pytest.fixture
    def mock_events(self):
        """Mock Events für Tests."""
        return [
            {"ts": 1000.0, "type": "probe_req", "client": "aa:bb:cc:dd:ee:00"},
            {"ts": 1001.0, "type": "probe_req", "client": "aa:bb:cc:dd:ee:01"},
        ]
    
    @pytest.fixture
    def mock_console(self):
        """Mock Console für Tests."""
        console = MagicMock()
        console.print = MagicMock()
        return console
    
    @pytest.fixture
    def temp_outdir(self, tmp_path):
        """Temporäres Ausgabeverzeichnis."""
        return tmp_path / "output"
    
    def test_plugin_metadata(self, plugin):
        """Test Plugin-Metadaten."""
        metadata = plugin.get_metadata()
        assert metadata.name == "Reinforcement Learning"
        assert metadata.version == "1.0.0"
        assert "RL" in metadata.description
        assert "gym" in metadata.dependencies
        assert "torch" in metadata.dependencies
    
    def test_action_type_enum(self):
        """Test ActionType Enum."""
        assert ActionType.SCAN_CHANNEL_1.value == 0
        assert ActionType.SCAN_CHANNEL_6.value == 1
        assert ActionType.PAUSE_SCANNING.value == 7
        assert ActionType.INCREASE_SCAN_RATE.value == 8
        assert ActionType.DECREASE_SCAN_RATE.value == 9
    
    def test_wifi_environment_state(self):
        """Test WiFiEnvironmentState Dataclass."""
        state = WiFiEnvironmentState(
            current_channel=6,
            channels_scanned=[1, 6, 11],
            aps_found=5,
            clients_found=10,
            signal_strength=-50.0,
            scan_duration=1.0,
            cpu_usage=0.5,
            memory_usage=0.3,
            last_scan_time=1000.0,
            total_events=100
        )
        
        assert state.current_channel == 6
        assert state.aps_found == 5
        assert state.clients_found == 10
        assert state.signal_strength == -50.0
    
    def test_wifi_scanning_environment_creation(self, plugin, mock_state, mock_events):
        """Test WiFi Scanning Environment Erstellung."""
        env = plugin._WiFiScanningEnvironment(mock_state, mock_events)
        
        assert env.current_step == 0
        assert env.max_steps == 1000
        assert len(env.channels) == 7  # [1, 6, 11, 36, 40, 44, 48]
        assert env.current_channel == 1
        assert len(env.scan_history) == 0
        
        # Performance Metrics
        assert env.performance_metrics['total_aps_found'] == 0
        assert env.performance_metrics['total_clients_found'] == 0
        assert env.performance_metrics['scan_efficiency'] == 0.0
    
    def test_environment_reset(self, plugin, mock_state, mock_events):
        """Test Environment Reset."""
        env = plugin._WiFiScanningEnvironment(mock_state, mock_events)
        
        # Simuliere einige Schritte
        env.current_step = 100
        env.current_channel = 6
        env.scan_history.append(1)
        env.scan_history.append(6)
        env.performance_metrics['total_aps_found'] = 5
        
        # Reset
        obs = env.reset()
        
        assert env.current_step == 0
        assert env.current_channel == 1
        assert len(env.scan_history) == 0
        assert env.performance_metrics['total_aps_found'] == 0
        assert len(obs) == 10  # Observation space size
    
    def test_environment_observation(self, plugin, mock_state, mock_events):
        """Test Environment Observation."""
        env = plugin._WiFiScanningEnvironment(mock_state, mock_events)
        obs = env._get_observation()
        
        assert len(obs) == 10
        assert all(0 <= val <= 1 for val in obs)  # Normalisierte Werte
        assert obs[0] == 1.0 / 48.0  # current_channel normalisiert
        assert obs[1] == 0.0  # total_aps_found normalisiert
        assert obs[2] == 0.0  # total_clients_found normalisiert
    
    def test_environment_step(self, plugin, mock_state, mock_events):
        """Test Environment Step."""
        env = plugin._WiFiScanningEnvironment(mock_state, mock_events)
        obs = env.reset()
        
        # Führe einen Schritt aus
        action = ActionType.SCAN_CHANNEL_6.value
        next_obs, reward, done, info = env.step(action)
        
        assert len(next_obs) == 10
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert env.current_step == 1
        assert env.current_channel == 6  # Kanal 6 gescannt
        assert 6 in env.scan_history
    
    def test_environment_action_execution(self, plugin, mock_state, mock_events):
        """Test Action Execution."""
        env = plugin._WiFiScanningEnvironment(mock_state, mock_events)
        
        # Test Kanal-Scanning
        reward = env._execute_action(ActionType.SCAN_CHANNEL_6.value)
        assert isinstance(reward, float)
        assert env.current_channel == 6
        assert 6 in env.scan_history
        
        # Test Pause
        reward = env._execute_action(ActionType.PAUSE_SCANNING.value)
        assert reward < 0  # Strafe für Pausieren
        
        # Test Scan-Rate Erhöhung
        reward = env._execute_action(ActionType.INCREASE_SCAN_RATE.value)
        assert reward > 0  # Belohnung für Aktivität
    
    def test_simple_rl_agent_creation(self, plugin):
        """Test Simple RL Agent Erstellung."""
        agent = plugin._SimpleRLAgent(state_size=10, action_size=10)
        
        assert agent.state_size == 10
        assert agent.action_size == 10
        assert agent.epsilon == 1.0
        assert agent.epsilon_min == 0.01
        assert agent.epsilon_decay == 0.995
        assert len(agent.episode_rewards) == 0
        assert len(agent.episode_lengths) == 0
    
    def test_simple_rl_agent_act(self, plugin):
        """Test Simple RL Agent Action Selection."""
        agent = plugin._SimpleRLAgent(state_size=10, action_size=10)
        
        state = np.random.random(10)
        
        # Test Exploration (mit hohem Epsilon)
        agent.epsilon = 1.0
        action = agent.act(state, training=True)
        assert 0 <= action < 10
        
        # Test Exploitation (mit niedrigem Epsilon)
        agent.epsilon = 0.0
        action = agent.act(state, training=True)
        assert 0 <= action < 10
    
    def test_simple_rl_agent_learn(self, plugin):
        """Test Simple RL Agent Learning."""
        agent = plugin._SimpleRLAgent(state_size=10, action_size=10)
        
        state = np.random.random(10)
        action = 0
        reward = 1.0
        next_state = np.random.random(10)
        done = False
        
        # Q-Table sollte leer sein
        state_key = tuple(state.round(2))
        assert all(agent.q_table[state_key] == 0)
        
        # Lerne
        agent.learn(state, action, reward, next_state, done)
        
        # Q-Table sollte aktualisiert sein
        assert agent.q_table[state_key][action] != 0
    
    def test_simple_rl_agent_epsilon_decay(self, plugin):
        """Test Epsilon Decay."""
        agent = plugin._SimpleRLAgent(state_size=10, action_size=10)
        
        initial_epsilon = agent.epsilon
        assert initial_epsilon == 1.0
        
        # Simuliere viele Lernschritte
        for _ in range(1000):
            state = np.random.random(10)
            action = 0
            reward = 0.0
            next_state = np.random.random(10)
            done = False
            agent.learn(state, action, reward, next_state, done)
        
        # Epsilon sollte reduziert sein
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_min
    
    def test_plugin_run_with_sufficient_data(self, plugin, mock_state, mock_events, mock_console, temp_outdir):
        """Test Plugin-Ausführung mit ausreichenden Daten."""
        temp_outdir.mkdir(exist_ok=True)
        
        with patch('joblib.dump') as mock_dump:
            with patch('builtins.open', create=True) as mock_open:
                plugin.run(mock_state, mock_events, mock_console, temp_outdir)
        
        # Überprüfe, dass Console-Ausgaben gemacht wurden
        assert mock_console.print.called
        
        # Überprüfe, dass Ergebnisse gespeichert wurden
        mock_dump.assert_called()
        mock_open.assert_called()
    
    def test_plugin_run_without_pytorch(self, plugin, mock_state, mock_events, mock_console, temp_outdir):
        """Test Plugin-Ausführung ohne PyTorch (Fallback zu Simple RL)."""
        temp_outdir.mkdir(exist_ok=True)
        
        with patch('plugins.reinforcement_learning.plugin.torch', None):
            with patch('joblib.dump') as mock_dump:
                with patch('builtins.open', create=True) as mock_open:
                    plugin.run(mock_state, mock_events, mock_console, temp_outdir)
        
        # Überprüfe, dass Console-Ausgaben gemacht wurden
        assert mock_console.print.called
        
        # Überprüfe, dass Simple RL Agent verwendet wurde
        console_calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Q-Learning" in call for call in console_calls)
    
    def test_training_results_structure(self, plugin, mock_state, mock_events):
        """Test Training Results Struktur."""
        env = plugin._WiFiScanningEnvironment(mock_state, mock_events)
        agent = plugin._SimpleRLAgent(state_size=10, action_size=10)
        
        training_results = plugin._train_rl_agent(env, agent, episodes=5)
        
        assert 'episode_rewards' in training_results
        assert 'episode_lengths' in training_results
        assert 'final_epsilon' in training_results
        assert len(training_results['episode_rewards']) == 5
        assert len(training_results['episode_lengths']) == 5
        assert isinstance(training_results['final_epsilon'], float)
    
    def test_agent_save_load(self, plugin, temp_outdir):
        """Test Agent Save/Load Funktionalität."""
        agent = plugin._SimpleRLAgent(state_size=10, action_size=10)
        
        # Simuliere einige Lernschritte
        for _ in range(10):
            state = np.random.random(10)
            action = 0
            reward = 1.0
            next_state = np.random.random(10)
            done = False
            agent.learn(state, action, reward, next_state, done)
        
        # Speichere Agent
        temp_outdir.mkdir(exist_ok=True)
        agent_file = temp_outdir / "test_agent.joblib"
        agent.save(agent_file)
        
        # Erstelle neuen Agent und lade
        new_agent = plugin._SimpleRLAgent(state_size=10, action_size=10)
        new_agent.load(agent_file)
        
        # Überprüfe, dass Q-Table geladen wurde
        assert len(new_agent.q_table) > 0
        assert new_agent.epsilon < 1.0  # Epsilon sollte reduziert sein
    
    def test_environment_simulate_scan_results(self, plugin, mock_state, mock_events):
        """Test Simulate Scan Results."""
        env = plugin._WiFiScanningEnvironment(mock_state, mock_events)
        
        # Test mit verschiedenen Kanälen
        for channel in [1, 6, 11, 36]:
            aps_found, clients_found = env._simulate_scan_results(channel)
            
            assert aps_found >= 0
            assert clients_found >= 0
            assert isinstance(aps_found, int)
            assert isinstance(clients_found, int)
    
    def test_environment_performance_metrics_update(self, plugin, mock_state, mock_events):
        """Test Performance Metrics Update."""
        env = plugin._WiFiScanningEnvironment(mock_state, mock_events)
        
        # Simuliere mehrere Scans
        env._execute_action(ActionType.SCAN_CHANNEL_1.value)
        env._execute_action(ActionType.SCAN_CHANNEL_6.value)
        env._execute_action(ActionType.SCAN_CHANNEL_11.value)
        
        # Überprüfe Performance Metrics
        assert env.performance_metrics['total_aps_found'] >= 0
        assert env.performance_metrics['total_clients_found'] >= 0
        assert 0 <= env.performance_metrics['scan_efficiency'] <= 1
        assert len(env.performance_metrics['channel_utilization']) > 0