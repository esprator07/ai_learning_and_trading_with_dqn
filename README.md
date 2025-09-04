# ai_learning_and_trading_with_dqn
Deep Q-Network (DQN) based trading agent trained on real market data (SOL/USDT). Uses features like RSI, MACD, and custom volume indicators to learn profitable trading strategies.

DQN Trader – Reinforcement Learning for Financial Trading

This repository contains an experimental implementation of a Deep Q-Network (DQN) agent trained to perform trading decisions using real historical market data.
The project aims to answer the question:

“Can we teach an AI agent how to trade profitably using reinforcement learning?”

The agent is trained on SOL/USDT market data collected between June and August (100 days), and learns to take trading actions (long, short, hold) based on hand-crafted features derived from technical indicators.
⚙️ Features Used

The model was trained using a set of engineered features, combining standard technical indicators with custom metrics:

return2_volume → Percentage difference between the current candle’s volume and the average volume of the last 30 candles.

rsi_14 → Relative Strength Index (14 periods).

macd_hist_norm → Normalized MACD histogram value.

macd_cross → MACD signal line cross indicator.

macd_hist_slope_norm → Normalized slope (change rate) of the MACD histogram.

🧠 Training Setup

Environment: Custom trading environment simulating a market with SOL/USDT data.

Algorithm: Deep Q-Network (DQN).

Rewards:

Positive reward for profitable trade closures.

Penalties for:

Closing a trade at a loss

Opening invalid trades (e.g., opening while already in position)

Closing without being in a position

Small gains/losses (±0.3%) → to discourage micro-trades

This design leads to negative cumulative rewards, since the agent is penalized in more cases than rewarded.
However, the balance evolution shows a clear profit growth, meaning the model is learning to trade in a way that maximizes portfolio balance despite heavy penalties.

📊 Best Result

Initial balance: $1000

Final balance: $1735 (+73.47%)

Total trades: 1412 (1206 long / 206 short)

Win rate: 52.5%

Key Plots

Balance Evolution → Agent steadily increases balance.

Cumulative Reward → Negative due to strict penalty scheme.

PnL Distribution → Most trades are small losses/gains, with a few larger profitable ones.

Win Rate Evolution → Stabilizes around break-even, but balance grows due to trade sizing.

(see data/ folder for full analysis plots and training metrics)

⚠️ Disclaimer

This project is for educational and research purposes only.
It is not financial advice. The results are experimental and should not be considered as real-world trading strategies.
