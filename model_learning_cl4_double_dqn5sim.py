import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os
import matplotlib


matplotlib.use("Agg")

# CPU multi-threading
torch.set_num_threads(6)
os.environ['OMP_NUM_THREADS'] = '6'

class DoubleDQN(nn.Module):
    """Double DQN Network Architecture"""
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], output_size=4):
        super(DoubleDQN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ImprovedDelayedBuffer:
    """
    Gelişmiş Delayed Reward Buffer
    Trade sırasında experiences'ı pending'de tutar
    Trade kapanınca calculated rewards ile buffer'a commit eder
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.pending_experiences = {}  # trade_id -> [experiences]
        
    def start_trade(self, trade_id):
        """Yeni trade başlatır"""
        self.pending_experiences[trade_id] = []
    
    def add_pending_experience(self, trade_id, state, action, next_state, done):
        """Trade sırasında experience'ları pending'de tutar"""
        if trade_id not in self.pending_experiences:
            self.pending_experiences[trade_id] = []
        
        self.pending_experiences[trade_id].append({
            'state': state.copy(),
            'action': action,
            'next_state': next_state.copy() if not done else np.zeros_like(state),
            'done': done
        })
    
    def add_immediate_experience(self, state, action, reward, next_state, done):
        """Trade dışındaki experience'ları hemen buffer'a ekler"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def commit_trade(self, trade_id, realized_pnl, fee_percent=0.05):
        """
        Trade kapandığında pending experiences'ları calculated rewards ile buffer'a commit eder
        """
        if trade_id not in self.pending_experiences:
            return
            
        experiences = self.pending_experiences[trade_id]
        if not experiences:
            return
            
        # Trade süresi
        trade_duration = len(experiences)
        
        # Fee maliyeti ve net PnL
        total_fee_cost = fee_percent * 2
        net_pnl = realized_pnl - total_fee_cost
        
        # Ödül hesaplamaları
        opening_reward = net_pnl * 0.25
        holding_reward_per_step = (net_pnl * 0.4) / max(trade_duration - 1, 1) if trade_duration > 1 else 0
        closing_reward = net_pnl 
        
        # Experience'ları calculated rewards ile buffer'a commit et
        for i, exp in enumerate(experiences):
            if i == 0:  # Opening
                calculated_reward = opening_reward
            elif i == trade_duration - 1:  # Closing
                calculated_reward = closing_reward
            else:  # Holding
                calculated_reward = holding_reward_per_step
            
            # Buffer'a ekle
            self.buffer.append({
                'state': exp['state'],
                'action': exp['action'],
                'reward': calculated_reward,
                'next_state': exp['next_state'],
                'done': exp['done']
            })
        
        # Pending'den çıkar
        del self.pending_experiences[trade_id]
        
        print(f"✅ Trade {trade_id} committed: PnL={realized_pnl:.2f}%, Net={net_pnl:.2f}%, "
              f"Steps={trade_duration}, Opening={opening_reward:.3f}, "
              f"Holding={holding_reward_per_step:.4f}x{trade_duration-1}")
    
    def sample(self, batch_size):
        """Batch sampling"""
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in batch:
            states.append(exp['state'])
            actions.append(exp['action'])
            rewards.append(exp['reward'])
            next_states.append(exp['next_state'])
            dones.append(exp['done'])
        
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(np.array(actions)),
                torch.FloatTensor(np.array(rewards)),
                torch.FloatTensor(np.array(next_states)),
                torch.BoolTensor(np.array(dones)))
    
    def __len__(self):
        return len(self.buffer)

class AdvancedTradingEnv(gym.Env):
    """
    Gelişmiş Bitcoin Futures Trading Environment
    - Action Masking
    - Unrealized PnL Feedback  
    - Improved Delayed Rewards
    """
    def __init__(self, df, lookback_window=30, initial_balance=1000, position_size=1000):
        super(AdvancedTradingEnv, self).__init__()
        
        self.df = df.copy()
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.position_size = position_size
        
        # Market features
        self.features = ['return2_volume', 'rsi_14', 'macd_hist_norm', 'macd_cross', 'macd_hist_slope_norm']
        
        # Action space: 0=Hold, 1=Long, 2=Short, 3=Close
        self.action_space = spaces.Discrete(4)
        
        # State space
        market_features = len(self.features) * lookback_window  # 150
        position_features = 5 * lookback_window                 # 150
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(market_features + position_features,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0  # 0=neutral, 1=long, -1=short
        self.entry_price = 0
        self.entry_step = 0
        self.steps_in_position = 0
        self.last_action = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.current_trade_id = None
        
        self.trades_history = []
        
        return self._get_state(), {}
    
    def get_valid_actions(self):
        """Action Masking: Pozisyona göre geçerli aksiyonları döndür"""
        if self.position == 0:  # Neutral position
            return [0, 1, 2]  # Hold, Long, Short
        else:  # In position
            return [0, 3]     # Hold, Close
    
    def get_action_mask(self):
        """Neural network için action mask"""
        mask = np.zeros(self.action_space.n, dtype=bool)
        valid_actions = self.get_valid_actions()
        mask[valid_actions] = True
        return mask
    
    def _get_state(self):
        """300 boyutlu state: market (150) + position (150)"""
        # Market features
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        window_data = self.df.iloc[start_idx:end_idx]
        
        market_state = []
        for feature in self.features:
            values = window_data[feature].values
            values = np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)
            values = np.clip(values, -100, 100)
            market_state.extend(values)
        
        # Position features
        current_price = self.df.iloc[self.current_step]['close']
        
        if self.position != 0 and self.entry_price > 0:
            entry_price_norm = (self.entry_price / current_price) - 1
            if self.position == 1:
                unrealized_pnl = ((current_price - self.entry_price) / self.entry_price) * 100
            else:
                unrealized_pnl = ((self.entry_price - current_price) / self.entry_price) * 100
        else:
            entry_price_norm = 0
            unrealized_pnl = 0
        
        position_features = [
            float(self.position) * 50,
            np.clip(entry_price_norm * 1000, -100, 100),
            np.clip(float(self.steps_in_position) / 5, -100, 100),
            float(self.last_action) * 25,
            np.clip(unrealized_pnl * 5, -100, 100)
        ]
        
        # Position features'ı 30 kez tekrarla
        position_state = []
        for _ in range(self.lookback_window):
            position_state.extend(position_features)
        
        full_state = market_state + position_state
        return np.array(full_state, dtype=np.float32)
    
    def step(self, action):
        """Environment step"""
        current_state = self._get_state()
        immediate_reward, trade_closed = self._process_action(action)
        
        self.current_step += 1
        self.last_action = action
        
        # Episode bitti mi?
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        if terminated and self.position != 0:
            # Force close
            close_reward, _ = self._process_action(3)
            immediate_reward += close_reward
        
        next_state = self._get_state() if not terminated else np.zeros_like(current_state)
        info = {'trade_closed': trade_closed} if trade_closed else {}
        
        return next_state, immediate_reward, terminated, truncated, info
    
    def _process_action(self, action):
        """Action processing with masking and unrealized PnL feedback"""
        current_price = self.df.iloc[self.current_step]['close']
        immediate_reward = 0
        trade_closed = False
        
        # Action masking kontrolü
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            return -10.0, False  # Invalid action penalty
        
        # Position tracking
        if self.position != 0:
            self.steps_in_position += 1
        
        # Unrealized PnL hesaplama
        unrealized_pnl = 0
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:  # Long
                unrealized_pnl = ((current_price - self.entry_price) / self.entry_price) * 100
            else:  # Short
                unrealized_pnl = ((self.entry_price - current_price) / self.entry_price) * 100
        
        # LONG AÇ
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            self.entry_step = self.current_step
            self.steps_in_position = 1
            self.current_trade_id = f"trade_{self.current_step}_{random.randint(1000,9999)}"
            immediate_reward = 0  # Açılışta ödül yok
            
        # SHORT AÇ
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price
            self.entry_step = self.current_step
            self.steps_in_position = 1
            self.current_trade_id = f"trade_{self.current_step}_{random.randint(1000,9999)}"
            immediate_reward = 0  # Açılışta ödül yok
            
        # POZİSYON KAPAT
        elif action == 3 and self.position != 0:
            realized_pnl = unrealized_pnl
            fee_cost = 0.05 * 2  # Fee maliyeti
            
            # Unrealized PnL penalty/bonus sistemi (sadece reward için)
            pnl_adjustment = self._calculate_pnl_adjustment(unrealized_pnl)
            
            # Balance güncelleme: Sadece gerçek PnL ve fee (adjustment HARİÇ)
            net_pnl_after_fee = realized_pnl #- fee_cost
            self.balance += (self.position_size * net_pnl_after_fee / 100)
            
            # Reward hesaplama: PnL + fee + adjustment 
            closing_reward = realized_pnl - fee_cost + pnl_adjustment
            immediate_reward = closing_reward
            
            # Trade kaydı
            self.total_trades += 1
            if realized_pnl > 0:
                self.profitable_trades += 1
                
            trade_record = {
                'trade_id': self.current_trade_id,
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'entry_time': self.df.iloc[self.entry_step]['open_time'].strftime('%Y-%m-%d %H:%M:%S'),  # ← EKLEYİN
                'exit_time': self.df.iloc[self.current_step]['open_time'].strftime('%Y-%m-%d %H:%M:%S'),   # ← EKLEYİN
                'position_type': 'LONG' if self.position == 1 else 'SHORT',
                'pnl_percent': realized_pnl,
                'fee_cost': fee_cost,
                'pnl_adjustment': pnl_adjustment,
                'net_pnl_after_fee': net_pnl_after_fee,
                'reward': closing_reward,
                'entry_step': self.entry_step,
                'exit_step': self.current_step,
                'duration': self.steps_in_position,
                'episode': getattr(self, 'current_episode', 0),  # ← EKLEYİN
                'balance_after_trade': self.balance,  # ← YENİ: Trade sonrası balance
                'episode_trade_number': self.total_trades + 1
            }
            self.trades_history.append(trade_record)
            
            # Debug: Trade bilgilerini göster
            adj_info = f" | PnL Adj: {pnl_adjustment:+.3f}" if abs(pnl_adjustment) > 0.001 else ""
            print(f"🎯 TRADE #{self.total_trades} CLOSED: {trade_record['position_type']} | "
                  f"Raw PnL: {realized_pnl:.2f}% | After Fee: {net_pnl_after_fee:.2f}% | "
                  f"Reward: {closing_reward:.2f}{adj_info} | "
                  f"Duration: {self.steps_in_position} | Balance: ${self.balance:.2f}")
            
            # Trade kapandı bilgisi
            trade_closed = {
                'trade_id': self.current_trade_id,
                'realized_pnl': realized_pnl,
                'fee_percent': 0.05
            }
            
            # Pozisyonu sıfırla
            self.position = 0
            self.entry_price = 0
            self.steps_in_position = 0
            self.current_trade_id = None
            
        # HOLD
        elif action == 0:
            if self.position != 0:
                # Unrealized PnL feedback (çok küçük)
                unrealized_feedback = np.sign(unrealized_pnl) * 0.001
                immediate_reward = unrealized_feedback
            else:
                immediate_reward = -0.000005  # Neutral holding penalty

        return immediate_reward, trade_closed
    
    def _calculate_pnl_adjustment(self, unrealized_pnl):
        """
        %0.3'ten küçük PnL'leri cezalandır, büyük PnL'leri ödüllendir
        """
        pnl_threshold = 0.3  # %0.3 minimum hedef
        
        if unrealized_pnl > 0:
            if unrealized_pnl < pnl_threshold:
                # Küçük kar cezası: %0.3'e yaklaştıkça azalır
                penalty_ratio = (pnl_threshold - unrealized_pnl) / pnl_threshold
                penalty = -1.5 * penalty_ratio  # Max -1.5 penalty
                
                # Debug için
                if penalty < -0.1:
                    print(f"  ⚠️  Small profit penalty: PnL {unrealized_pnl:.2f}% < {pnl_threshold}%, penalty: {penalty:.3f}")
                
                return penalty
            else:
                # Büyük kar bonusu: threshold'u geçtikten sonra bonus
                bonus_ratio = min((unrealized_pnl - pnl_threshold) / 2.0, 1.0)  # Max %2 için full bonus
                bonus = 0.5 * bonus_ratio  # Max +0.5 bonus
                
                if bonus > 0.1:
                    print(f"  ✅ Big profit bonus: PnL {unrealized_pnl:.2f}% > {pnl_threshold}%, bonus: {bonus:.3f}")
                
                return bonus
        
        elif unrealized_pnl < 0:
            # Loss durumunda
            if unrealized_pnl > -0.5:
                # Quick stop-loss teşviki (küçük loss)
                quick_stop_bonus = -0.2
                print(f"  🛡️ Quick stop-loss ceza: PnL {unrealized_pnl:.2f}%, ceza: {quick_stop_bonus:.3f}")
                return quick_stop_bonus
            else:
                # Büyük loss - sadece fee cezası yeterli
                return 0
        
        return 0  # Sıfır PnL
    
    def _get_episode_info(self):
        """Episode sonu bilgileri"""
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.profitable_trades / max(self.total_trades, 1)) * 100
        
        return {
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_balance': self.balance,
            'trades_history': self.trades_history
        }

class AdvancedDQNAgent:
    """
    Gelişmiş DQN Agent - Action Masking + Improved Delayed Rewards
    """
    def __init__(self, state_size, action_size, lr=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ImprovedDelayedBuffer(100000)
        self.epsilon = 1.0
        self.epsilon_min = 0.00
        self.epsilon_decay = 0.999992
        self.learning_rate = lr
        self.gamma = 0.95
        self.update_freq = 1000
        
        # Networks
        self.device = torch.device("cpu")
        self.q_network = DoubleDQN(state_size, [512, 256, 128], action_size).to(self.device)
        self.target_network = DoubleDQN(state_size, [512, 256, 128], action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.update_target_network()
        self.step_count = 0
        
    def update_target_network(self):
        """Target network güncelle"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember_pending(self, state, action, next_state, done, trade_id):
        """Pending experience kaydet"""
        self.memory.add_pending_experience(trade_id, state, action, next_state, done)
    
    def remember_immediate(self, state, action, reward, next_state, done):
        """Immediate experience kaydet"""
        self.memory.add_immediate_experience(state, action, reward, next_state, done)
    
    def start_trade(self, trade_id):
        """Trade başlat"""
        self.memory.start_trade(trade_id)
    
    def commit_trade(self, trade_id, realized_pnl, fee_percent=0.05):
        """Trade commit et"""
        self.memory.commit_trade(trade_id, realized_pnl, fee_percent)
    
    def act(self, state, action_mask=None, training=True):
        """Action selection with masking"""
        if training and np.random.random() <= self.epsilon:
            # Masked random action
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                return np.random.choice(valid_actions)
            else:
                return random.randrange(self.action_size)
        
        # Neural network prediction
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor).cpu().data.numpy()[0]
        
        # Apply action mask
        if action_mask is not None:
            q_values[~action_mask] = -np.inf
        
        return np.argmax(q_values)
    
    def replay(self, batch_size=32):
        """Model training"""
        batch = self.memory.sample(batch_size)
        if batch is None:
            return
        
        states, actions, rewards, next_states, dones = batch
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN
        next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions).detach()
        
        # Target values
        target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Target network update
        self.step_count += 1
        if self.step_count % self.update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def save(self, filepath):
        """Model kaydet"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load(self, filepath):
        """Model yükle"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.step_count = checkpoint.get('step_count', 0)

class AdvancedTrainer:
    """
    Gelişmiş DQN Training Manager
    """

    def __init__(self, data_file):
        print("📂 Veri yükleniyor...")
        self.df = pd.read_csv(data_file)
        self.df['open_time'] = pd.to_datetime(self.df['open_time'])
        
        print(f"✅ {len(self.df):,} mum yüklendi")
        print(f"📅 Tarih: {self.df['open_time'].min()} - {self.df['open_time'].max()}")
        
        # Environment ve Agent
        self.env = AdvancedTradingEnv(self.df)
        self.agent = AdvancedDQNAgent(
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            lr=0.0005
        )
        
        self.training_history = []
        self.episode_balance_tracking = []  # Episode içi balance takibi için
    

    def train(self, episodes=1000, max_steps_per_episode=None):
        """Advanced DQN Training"""
        all_trades = []  

        if max_steps_per_episode is None:
            max_steps_per_episode = len(self.df) - 35
        
        print(f"\n🚀 ADVANCED DQN TRAINING")
        print("=" * 50)
        print(f"📊 Episodes: {episodes}")
        print(f"🎮 State boyutu: {self.env.observation_space.shape[0]}")
        print(f"🎲 Action boyutu: {self.env.action_space.n}")
        print(f"✅ Action masking aktif")
        print(f"🔄 Improved delayed reward sistemi")
        print(f"📈 Unrealized PnL feedback")
        
        best_score = -np.inf
        scores_deque = deque(maxlen=100)
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            self.env.current_episode = episode
            total_reward = 0
            steps = 0
            losses = []
            invalid_actions = 0
            
            # Episode balance tracking
            episode_balances = [self.env.balance]  # Starting balance
            episode_rewards = [0]  # Starting reward
            episode_steps = [0]
            episode_trades_detail = []
            start_balance = self.env.balance
            cumulative_reward = 0
            
            for step in range(max_steps_per_episode):
                # Action masking
                action_mask = self.env.get_action_mask()
                action = self.agent.act(state, action_mask, training=True)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Experience handling
                trade_id = self.env.current_trade_id
                
                # Yeni trade başladıysa
                if action in [1, 2] and trade_id:
                    self.agent.start_trade(trade_id)
                
                # Experience kaydetme
                if trade_id and action in [0, 1, 2]:  # Trade içinde
                    self.agent.remember_pending(state, action, next_state, done, trade_id)
                else:  # Trade dışı
                    self.agent.remember_immediate(state, action, reward, next_state, done)
                
                # Invalid action tracking
                if reward == -10.0:
                    invalid_actions += 1
                
                # Trade kapandıysa commit et
                if info.get('trade_closed'):
                    trade_info = info['trade_closed']
                    self.agent.commit_trade(
                        trade_info['trade_id'], 
                        trade_info['realized_pnl'], 
                        trade_info['fee_percent']
                    )
                    # Trade detayını kaydet
                    if len(self.env.trades_history) > 0:
                        latest_trade = self.env.trades_history[-1].copy()
                        latest_trade['episode'] = episode  # ← BU SATIRI EKLEYİN
                        episode_trades_detail.append(latest_trade)
                        all_trades.append(latest_trade)  # ← BU SATIRI EKLEYİN
                
                state = next_state
                total_reward += reward
                cumulative_reward += reward
                steps += 1
                
                # Balance tracking (her 50 step'te bir)
                if steps % 50 == 0:
                    episode_balances.append(self.env.balance)
                    episode_rewards.append(cumulative_reward)
                    episode_steps.append(steps)
                
                # Training
                if len(self.agent.memory) > 1000:
                    loss = self.agent.replay(batch_size=32)
                    if loss is not None:
                        losses.append(loss)
                
                if done:
                    break
            
            scores_deque.append(total_reward)
            avg_score = np.mean(scores_deque)
            
            # Save best model
            if avg_score > best_score and episode > 10:
                best_score = avg_score
                self.agent.save(f"best_advanced_dqn_{best_score:.2f}.pth")
                print(f"🏆 Yeni en iyi model! Score: {best_score:.2f}")
            
            # Progress reporting
            if episode % 5 == 0 or episode < 10:
                avg_loss = np.mean(losses) if losses else 0
                win_rate = (self.env.profitable_trades / max(self.env.total_trades, 1)) * 100
                
                print(f"Episode {episode:4d} | Score: {total_reward:8.2f} | "
                      f"Avg: {avg_score:8.2f} | Eps: {self.agent.epsilon:.3f} | "
                      f"Trades: {self.env.total_trades} | Win: {win_rate:.1f}% | "
                      f"Invalid: {invalid_actions} | Buffer: {len(self.agent.memory)}")
            
            # Store metrics
            episode_metrics = {
                'episode': episode,
                'total_reward': total_reward,
                'avg_reward': avg_score,
                'epsilon': self.agent.epsilon,
                'trades': self.env.total_trades,
                'win_rate': (self.env.profitable_trades / max(self.env.total_trades, 1)) * 100,
                'invalid_actions': invalid_actions,
                'buffer_size': len(self.agent.memory),
                'episode_balance': self.env.balance,  # Episode sonu balance
                'balance_change': self.env.balance - start_balance  # Episode balance değişimi
            }
            self.training_history.append(episode_metrics)
            
            # Belirli episode'ları balance tracking için kaydet
            if episode % 50 == 0 or episode < 5 or episode in [episodes//4, episodes//2, 3*episodes//4]:
                self.episode_balance_tracking.append({
                    'episode': episode,
                    'steps': episode_steps,
                    'balances': episode_balances,
                    'rewards': episode_rewards,
                    'trades': episode_trades_detail,
                    'start_step': 0
                })
            
            # Her episode'un detaylı analizini kaydet (ilk 10 ve her 25'te bir)
            if episode < 10 or episode % 25 == 0:
                self._save_episode_analysis(episode, episode_steps, episode_balances, 
                                          episode_rewards, episode_trades_detail, 
                                          start_balance, self.env)
        
        # Final save
        self.agent.save(f"final_advanced_dqn_{episodes}.pth")
        print(f"✅ Training tamamlandı! Final model kaydedildi.")
        
        self._save_trading_data(all_trades, episodes)

        # Plot training results
        self._plot_training_results()
        
        # Plot episode balance evolutions
        if self.episode_balance_tracking:
            self._plot_episode_balance_evolution()
        
        return self.training_history
    
    def _plot_training_results(self):
        """Training sonuçlarını görselleştir"""
        if not self.training_history:
            return
        
        df_metrics = pd.DataFrame(self.training_history)
        
        # 2x3 subplot (Balance grafiği eklendi)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Rewards Evolution
        ax1.plot(df_metrics['episode'], df_metrics['total_reward'], alpha=0.3, color='lightblue', label='Episode Reward')
        ax1.plot(df_metrics['episode'], df_metrics['avg_reward'], linewidth=2, color='darkblue', label='Rolling Avg (100)')
        ax1.set_title('Training Rewards Evolution', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Balance Evolution
        initial_balance = 1000
        ax2.plot(df_metrics['episode'], df_metrics['episode_balance'], linewidth=2, color='green', label='Episode Balance')
        ax2.axhline(y=initial_balance, color='red', linestyle='--', alpha=0.7, label=f'Initial Balance ({initial_balance})')
        
        # Balance change percentage
        balance_change_pct = ((df_metrics['episode_balance'] - initial_balance) / initial_balance) * 100
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df_metrics['episode'], balance_change_pct, 'orange', alpha=0.7, label='Balance Change %')
        
        ax2.set_title('Portfolio Balance Evolution', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Balance ($)', color='green')
        ax2_twin.set_ylabel('Change (%)', color='orange')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Win Rate Evolution  
        ax3.plot(df_metrics['episode'], df_metrics['win_rate'], 'g-', linewidth=2, label='Win Rate %')
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Break-even (50%)')
        ax3.set_title('Win Rate Evolution', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Win Rate %')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # 4. Trading Activity
        ax4.plot(df_metrics['episode'], df_metrics['trades'], 'orange', linewidth=2, label='Trades per Episode')
        ax4.set_title('Trading Activity', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Number of Trades')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Invalid Actions & Buffer Size
        ax5_twin = ax5.twinx()
        line1 = ax5.plot(df_metrics['episode'], df_metrics['invalid_actions'], 'r-', alpha=0.7, label='Invalid Actions')
        line2 = ax5_twin.plot(df_metrics['episode'], df_metrics['buffer_size'], 'purple', alpha=0.7, label='Buffer Size')
        
        ax5.set_title('Invalid Actions & Memory Growth', fontweight='bold', fontsize=14)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Invalid Actions', color='red')
        ax5_twin.set_ylabel('Buffer Size', color='purple')
        ax5.tick_params(axis='y', labelcolor='red')
        ax5_twin.tick_params(axis='y', labelcolor='purple')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # 6. Epsilon Decay & Performance Correlation
        ax6_twin = ax6.twinx()
        line1 = ax6.plot(df_metrics['episode'], df_metrics['epsilon'], 'b-', linewidth=2, label='Epsilon (Exploration)')
        line2 = ax6_twin.plot(df_metrics['episode'], df_metrics['balance_change'], 'gold', alpha=0.8, label='Episode Balance Change')
        
        ax6.set_title('Exploration vs Balance Performance', fontweight='bold', fontsize=14)
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Epsilon', color='b')
        ax6_twin.set_ylabel('Balance Change ($)', color='gold')
        ax6.tick_params(axis='y', labelcolor='b')
        ax6_twin.tick_params(axis='y', labelcolor='gold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper right')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'advanced_dqn_training_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"💾 Training grafikleri kaydedildi: {filename}")
        
        # Training özeti yazdır
        final_balance = df_metrics['episode_balance'].iloc[-1]
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        max_balance = df_metrics['episode_balance'].max()
        min_balance = df_metrics['episode_balance'].min()
        
        print(f"\n📊 TRAINING ÖZETİ:")
        print(f"📈 En yüksek avg reward: {df_metrics['avg_reward'].max():.2f}")
        print(f"📈 Son avg reward: {df_metrics['avg_reward'].iloc[-1]:.2f}")
        print(f"🎯 En yüksek win rate: {df_metrics['win_rate'].max():.1f}%")
        print(f"🎯 Son win rate: {df_metrics['win_rate'].iloc[-1]:.1f}%")
        print(f"💰 Starting balance: ${initial_balance:,.0f}")
        print(f"💰 Final balance: ${final_balance:,.0f}")
        print(f"💰 Total return: {total_return:+.2f}%")
        print(f"💰 Max balance: ${max_balance:,.0f}")
        print(f"💰 Min balance: ${min_balance:,.0f}")
        print(f"📊 Ortalama trades per episode: {df_metrics['trades'].mean():.1f}")
        print(f"❌ Toplam invalid actions: {df_metrics['invalid_actions'].sum()}")
        print(f"🧠 Final buffer size: {df_metrics['buffer_size'].iloc[-1]}")

    def _plot_episode_balance_evolution(self):
        """
        Seçili episode'ların balance evolution'ını çizer
        """
        if not self.episode_balance_tracking:
            return
            
        # Episode sayısına göre subplot ayarla
        n_episodes = len(self.episode_balance_tracking)
        if n_episodes <= 4:
            rows, cols = 2, 2
        elif n_episodes <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
            
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        if n_episodes == 1:
            axes = [axes]
        axes = axes.flatten() if n_episodes > 1 else axes
        
        for i, episode_data in enumerate(self.episode_balance_tracking[:rows*cols]):
            ax = axes[i] if n_episodes > 1 else axes
            
            ep_num = episode_data['episode']
            steps = episode_data['steps']
            balances = episode_data['balances']
            trades = episode_data['trades']
            
            # Balance line
            ax.plot(steps, balances, linewidth=2, color='darkblue', alpha=0.8)
            ax.fill_between(steps, balances, alpha=0.2, color='lightblue')
            
            # Trade noktalarını işaretle
            for trade in trades:
                # Trade'in step pozisyonunu bul
                trade_step_approx = None
                for j, step in enumerate(steps):
                    if step >= (trade['exit_step'] - trade['entry_step']):
                        trade_step_approx = step
                        break
                
                if trade_step_approx and j < len(balances):
                    trade_balance = balances[j]
                    color = 'green' if trade['pnl_percent'] > 0 else 'red'
                    marker = '^' if trade['position_type'] == 'LONG' else 'v'  # Standard matplotlib markers
                    ax.scatter(trade_step_approx, trade_balance, 
                             color=color, s=80, alpha=0.8, marker=marker)
                    
                    # Trade bilgisini göster
                    ax.annotate(f"{trade['pnl_percent']:.1f}%", 
                               (trade_step_approx, trade_balance),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
            
            # Başlangıç ve bitiş balance
            start_balance = balances[0]
            end_balance = balances[-1]
            balance_change = end_balance - start_balance
            change_pct = (balance_change / start_balance) * 100
            
            ax.set_title(f'Episode {ep_num} Balance Evolution\n'
                        f'Start: ${start_balance:.0f} → End: ${end_balance:.0f} '
                        f'({change_pct:+.2f}%)', fontsize=10, fontweight='bold')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Balance ($)')
            ax.grid(True, alpha=0.3)
            
            # Y-axis range ayarla
            y_min = min(balances) * 0.95
            y_max = max(balances) * 1.05
            ax.set_ylim(y_min, y_max)
            
        # Boş subplot'ları gizle
        for j in range(n_episodes, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # Kaydet
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'episode_balance_evolution_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"💾 Episode balance evolution grafikleri kaydedildi: {filename}")
        
        # Episode balance özeti
        print(f"\n💰 EPISODE BALANCE ÖZETİ:")
        for episode_data in self.episode_balance_tracking:
            ep_num = episode_data['episode']
            balances = episode_data['balances']
            trades = episode_data['trades']
            
            start_bal = balances[0]
            end_bal = balances[-1]
            change_pct = ((end_bal - start_bal) / start_bal) * 100
            
            print(f"Episode {ep_num:3d}: ${start_bal:8.0f} → ${end_bal:8.0f} "
                  f"({change_pct:+6.2f}%) | Trades: {len(trades):2d}")
    
    def _save_episode_analysis(self, episode_num, steps, balances, rewards, trades, start_balance, env):
        """
        Her episode'ın detaylı analizini ayrı PNG olarak kaydet
        """
        if not steps or len(steps) < 2:
            return
        
        # Trade analizi
        long_trades = [t for t in trades if t['position_type'] == 'LONG']
        short_trades = [t for t in trades if t['position_type'] == 'SHORT']
        profitable_trades = [t for t in trades if t['pnl_percent'] > 0]
        
        # İstatistikler
        total_trades = len(trades)
        long_count = len(long_trades)
        short_count = len(short_trades)
        profitable_count = len(profitable_trades)
        win_rate = (profitable_count / max(total_trades, 1)) * 100
        
        final_balance = balances[-1] if balances else start_balance
        balance_change = final_balance - start_balance
        balance_change_pct = (balance_change / start_balance) * 100
        
        final_reward = rewards[-1] if rewards else 0
        
        # Figure oluştur
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Üst bilgi kutusu
        fig.suptitle(f'Episode {episode_num} - Detailed Analysis', fontsize=16, fontweight='bold')
        
        # Info text
        info_text = (f"Total Trades: {total_trades} | Long: {long_count} | Short: {short_count} | "
                    f"Win Rate: {win_rate:.1f}% | Balance: ${start_balance:.0f} → ${final_balance:.0f} "
                    f"({balance_change_pct:+.2f}%) | Final Reward: {final_reward:.2f}")
        fig.text(0.5, 0.95, info_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 1. Balance Evolution
        ax1.plot(steps, balances, linewidth=3, color='darkgreen', alpha=0.8)
        ax1.fill_between(steps, balances, alpha=0.2, color='lightgreen')
        
        # Trade noktalarını işaretle
        for i, trade in enumerate(trades):
            # Trade'in yaklaşık step pozisyonunu bul
            trade_step_idx = min(int((trade['exit_step'] - trade['entry_step']) / 50), len(steps) - 1)
            if trade_step_idx < len(balances):
                trade_balance = balances[trade_step_idx]
                color = 'green' if trade['pnl_percent'] > 0 else 'red'
                marker = '^' if trade['position_type'] == 'LONG' else 'v'
                size = min(100, 50 + abs(trade['pnl_percent']) * 10)  # Size based on PnL
                
                ax1.scatter(steps[trade_step_idx], trade_balance, 
                           color=color, s=size, alpha=0.8, marker=marker, edgecolors='black', linewidth=1)
                
                # PnL label
                ax1.annotate(f"{trade['pnl_percent']:.1f}%", 
                           (steps[trade_step_idx], trade_balance),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', alpha=0.9,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
        
        ax1.set_title('Balance Evolution with Trades', fontweight='bold')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Balance ($)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=start_balance, color='red', linestyle='--', alpha=0.7, label=f'Start Balance')
        ax1.legend()
        
        # 2. Reward Accumulation
        ax2.plot(steps, rewards, linewidth=3, color='blue', alpha=0.8)
        ax2.fill_between(steps, rewards, alpha=0.2, color='lightblue')
        ax2.set_title('Cumulative Reward Evolution', fontweight='bold')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Cumulative Reward')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 3. Trade Analysis (Bar Chart)
        categories = ['Total\nTrades', 'Long\nTrades', 'Short\nTrades', 'Profitable\nTrades']
        values = [total_trades, long_count, short_count, profitable_count]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        
        bars = ax3.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('Trading Statistics', fontweight='bold')
        ax3.set_ylabel('Count')
        
        # Bar üzerine değerleri yaz
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. PnL Distribution (Histogram)
        if trades:
            pnl_values = [t['pnl_percent'] for t in trades]
            
            # Histogram
            n_bins = min(10, max(3, len(trades) // 2))
            counts, bins, patches = ax4.hist(pnl_values, bins=n_bins, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Renklendirme (kırmızı loss, yeşil profit)
            for i, patch in enumerate(patches):
                bin_center = (bins[i] + bins[i+1]) / 2
                if bin_center < 0:
                    patch.set_facecolor('lightcoral')
                else:
                    patch.set_facecolor('lightgreen')
            
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax4.set_title('PnL Distribution', fontweight='bold')
            ax4.set_xlabel('PnL (%)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
            
            # İstatistik bilgileri
            avg_pnl = np.mean(pnl_values)
            max_pnl = max(pnl_values)
            min_pnl = min(pnl_values)
            
            stats_text = f'Avg: {avg_pnl:.2f}%\nMax: {max_pnl:.2f}%\nMin: {min_pnl:.2f}%'
            ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No Trades in This Episode', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14, fontweight='bold')
            ax4.set_title('PnL Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Üst bilgi için yer bırak
        
        # Kaydet
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'episode_{episode_num:03d}_analysis_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # Memory için figure'ı kapat
        
        # Sadece ilk birkaç episode için console message
        if episode_num < 5:
            print(f"📊 Episode {episode_num} analysis kaydedildi: {filename}")
    
    def _save_trading_data(self, all_trades, total_episodes):
        """Tüm trading verilerini JSON ve CSV olarak kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON formatında kaydet (detaylı)
        import json
        
        trading_data = {
            'training_info': {
                'timestamp': timestamp,
                'total_episodes': total_episodes,
                'total_trades': len(all_trades),
                'data_file': 'solusdt_futures_100days_with_macd.csv'
            },
            'trades': all_trades,
            'episode_summary': self.training_history
        }
        
        json_filename = f'trading_results_{timestamp}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(trading_data, f, indent=2, ensure_ascii=False, default=str)
        
        # CSV formatında kaydet (sadece trade'ler)
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            csv_filename = f'trades_{timestamp}.csv'
            trades_df.to_csv(csv_filename, index=False)
            
            print(f"💾 Trading verileri kaydedildi:")
            print(f"   📄 JSON: {json_filename}")
            print(f"   📊 CSV: {csv_filename}")
            print(f"   📈 Toplam trade: {len(all_trades)}")
            
            # Özet istatistikler
            profitable_trades = len([t for t in all_trades if t['pnl_percent'] > 0])
            total_pnl = sum([t['net_pnl_after_fee'] for t in all_trades])
            win_rate = (profitable_trades / len(all_trades)) * 100
            
            print(f"   🎯 Win rate: {win_rate:.1f}%")
            print(f"   💰 Toplam PnL: {total_pnl:.2f}%")
        else:
            print("⚠️ Kaydedilecek trade bulunamadı")

def main():
    """Ana fonksiyon"""
    data_file = "solusdt_futures_100days_with_macd.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Veri dosyası bulunamadı: {data_file}")
        return
    
    trainer = AdvancedTrainer(data_file)
    
    print(f"\n🎯 ADVANCED DQN SİSTEMİ")
    print("=" * 50)
    print(f"✅ Action masking - Invalid actions önlendi")
    print(f"🔄 Improved delayed reward - Timing sorunu çözüldü") 
    print(f"📈 Unrealized PnL feedback - Holding'te küçük sinyal")
    print(f"🎯 Açılış: PnL * 0.25")
    print(f"📊 Holding: PnL * 0.4 / holding_steps") 
    print(f"💰 Kapanış: PnL * 0.35")
    
    try:
        episodes = int(input("🔢 Kaç episode eğitim? (önerilen: 1000): "))
        if episodes <= 0:
            episodes = 1000
    except:
        episodes = 1000
    
    user_input = input(f"\n🚀 {episodes} episode eğitime başlamak için ENTER (q=çıkış): ")
    
    if user_input.lower() != 'q':
        print(f"\n⏰ Training başlıyor...")
        training_results = trainer.train(episodes=episodes)
        
        print(f"\n✅ TRAINING TAMAMLANDI!")
        final_episode = training_results[-1]
        print(f"📈 Final Score: {final_episode['avg_reward']:.2f}")
        print(f"🎯 Final Win Rate: {final_episode['win_rate']:.1f}%")
        print(f"📊 Final Trades: {final_episode['trades']}")
        
    else:
        print("❌ Training iptal edildi")

if __name__ == "__main__":
    main()