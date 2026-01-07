import pandas as pd

class TradingEnvironment:
    """
    A simple trading environment for Reinforcement Learning
    """

    def __init__(self, data_path, initial_balance=10000):
        # -----------------------------
        # Load market data
        # -----------------------------
        self.data = pd.read_csv(data_path)

        # -----------------------------
        # Environment variables
        # -----------------------------
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = 0

        # Portfolio value = cash + value of shares
        self.portfolio_value = initial_balance

    def reset(self):
        """
        Reset the environment to the starting state
        """
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.portfolio_value = self.initial_balance

        return self._get_state()

    def _get_state(self):
        """
        Get the current state (what the agent sees)
        """
        state = self.data.iloc[self.current_step][
            ["Close", "SMA_20", "SMA_50", "Volume"]
        ].values

        return state

    def step(self, action):
        """
        Take an action:
        0 = Hold
        1 = Buy
        2 = Sell
        """

        # Current price
        current_price = self.data.iloc[self.current_step]["Close"]

        # Save portfolio value before action
        prev_portfolio_value = self.portfolio_value

        trade_penalty = 0

        # -----------------------------
        # Execute action
        # -----------------------------
        if action == 1:  # BUY
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
                trade_penalty = -0.1

        elif action == 2:  # SELL
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
                trade_penalty = -0.1

        # -----------------------------
        # Move to next day
        # -----------------------------
        self.current_step += 1

        # -----------------------------
        # Update portfolio value
        # -----------------------------
        self.portfolio_value = (
            self.balance + self.shares_held * current_price
        )

        # -----------------------------
        # Reward calculation
        # -----------------------------
        reward = (self.portfolio_value - prev_portfolio_value) + trade_penalty

        # -----------------------------
        # Check if episode finished
        # -----------------------------
        done = self.current_step >= len(self.data) - 1

        # -----------------------------
        # Get next state
        # -----------------------------
        next_state = self._get_state()

        return next_state, reward, done
