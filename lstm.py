import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class lstm():
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #bobot input
        self.bobot_i_fg = np.random.randn(hidden_size, input_size)
        self.bobot_i_ig = np.random.randn(hidden_size, input_size)
        self.bobot_i_cc = np.random.randn(hidden_size, input_size)
        self.bobot_i_og = np.random.randn(hidden_size, input_size)

        #bobot short-term (hidden)
        self.bobot_h_fg = np.random.randn(hidden_size, hidden_size)
        self.bobot_h_ig = np.random.randn(hidden_size, hidden_size)
        self.bobot_h_cc = np.random.randn(hidden_size, hidden_size)
        self.bobot_h_og = np.random.randn(hidden_size, hidden_size)

        #bias
        self.bias_fg = np.zeros((hidden_size, 1))
        self.bias_ig = np.zeros((hidden_size, 1))
        self.bias_cc = np.zeros((hidden_size, 1))
        self.bias_og = np.zeros((hidden_size, 1))

        # Output layer
        self.bobot_y = np.random.randn(output_size, hidden_size)
        self.bias_y = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):  
        s = self.sigmoid(x)
        return s * (1 - s)

    def dtanh(self, x):   
        return 1 - np.tanh(x) ** 2

    def forward(self, input_sequence):
        #nilai untuk proses backpropagation
        self.cache = []

        prev_h = np.zeros((self.hidden_size, 1))
        prev_c = np.zeros((self.hidden_size, 1))

        output = []

        for i in range(len(input_sequence)):
            input = input_sequence[i].reshape(-1, 1)  

            fg = self.sigmoid(self.bobot_i_fg @ input + self.bobot_h_fg @ prev_h + self.bias_fg)
            ig = self.sigmoid(self.bobot_i_ig @ input + self.bobot_h_ig @ prev_h + self.bias_ig)
            cc = np.tanh(self.bobot_i_cc @ input + self.bobot_h_cc @ prev_h + self.bias_cc)
            og = self.sigmoid(self.bobot_i_og @ input + self.bobot_h_og @ prev_h + self.bias_og)

            c_prev = prev_c.copy()
            #update cell state (long term)
            c = fg * prev_c + ig * cc

            #update hidden state (short term)
            h = og * np.tanh(c)

            #output
            y = self.bobot_y @ h + self.bias_y
            output.append(y)

            self.cache.append((input, prev_h, fg, ig, og, cc, c_prev, c))

        return output[-1]
    
    def backward(self, gradient_output, learning_rate=0.1):
        # Initialize gradients
        dWi_fg = np.zeros_like(self.bobot_i_fg)
        dWi_ig = np.zeros_like(self.bobot_i_ig)
        dWi_cc = np.zeros_like(self.bobot_i_cc)
        dWi_og = np.zeros_like(self.bobot_i_og)

        dWh_fg = np.zeros_like(self.bobot_h_fg)
        dWh_ig = np.zeros_like(self.bobot_h_ig)
        dWh_cc = np.zeros_like(self.bobot_h_cc)
        dWh_og = np.zeros_like(self.bobot_h_og)

        db_fg = np.zeros_like(self.bias_fg)
        db_ig = np.zeros_like(self.bias_ig)
        db_cc = np.zeros_like(self.bias_cc)
        db_og = np.zeros_like(self.bias_og)

        dWy = np.zeros_like(self.bobot_y)
        dby = np.zeros_like(self.bias_y)

        # Initialize next-step gradients
        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))

        # Loop backwards through time
        for i in reversed(range(len(gradient_output))):
            x, h_prev, fg, ig, og, cc, c_prev, c = self.cache[i]

            # (1) Output layer gradient
            dy = gradient_output[i].reshape(-1, 1)
            dWy += dy @ self.cache[i][-1].T  # h = last element in cache (or use h from loop)
            dby += dy

            # (2) Gradient from output + next time step
            dh = self.bobot_y.T @ dy + dh_next
            dc = dh * og * self.dtanh(c) + dc_next

            # (3) Gate gradients
            dog = dh * np.tanh(c)
            dfg = dc * c_prev
            dig = dc * cc
            dcc = dc * ig

            # (4) Pre-activation gradients (apply derivative of activation)
            dog_input = dog * self.dsigmoid(self.bobot_i_og @ x + self.bobot_h_og @ h_prev + self.bias_og)
            dfg_input = dfg * self.dsigmoid(self.bobot_i_fg @ x + self.bobot_h_fg @ h_prev + self.bias_fg)
            dig_input = dig * self.dsigmoid(self.bobot_i_ig @ x + self.bobot_h_ig @ h_prev + self.bias_ig)
            dcc_input = dcc * self.dtanh(self.bobot_i_cc @ x + self.bobot_h_cc @ h_prev + self.bias_cc)

            # (5) Accumulate weight gradients
            dWi_fg += dfg_input @ x.T
            dWi_ig += dig_input @ x.T
            dWi_cc += dcc_input @ x.T
            dWi_og += dog_input @ x.T

            dWh_fg += dfg_input @ h_prev.T
            dWh_ig += dig_input @ h_prev.T
            dWh_cc += dcc_input @ h_prev.T
            dWh_og += dog_input @ h_prev.T

            db_fg += dfg_input
            db_ig += dig_input
            db_cc += dcc_input
            db_og += dog_input

            # (6) Compute gradients to propagate to previous time step
            dx = (self.bobot_i_fg.T @ dfg_input +
                  self.bobot_i_ig.T @ dig_input +
                  self.bobot_i_cc.T @ dcc_input +
                  self.bobot_i_og.T @ dog_input)

            dh_next = (self.bobot_h_fg.T @ dfg_input +
                       self.bobot_h_ig.T @ dig_input +
                       self.bobot_h_cc.T @ dcc_input +
                       self.bobot_h_og.T @ dog_input)

            dc_next = dc * fg  # ← key! forget gate modulates how much c_prev matters

        # (7) Update parameters
        self.bobot_i_fg -= learning_rate * dWi_fg
        self.bobot_i_ig -= learning_rate * dWi_ig
        self.bobot_i_cc -= learning_rate * dWi_cc
        self.bobot_i_og -= learning_rate * dWi_og

        self.bobot_h_fg -= learning_rate * dWh_fg
        self.bobot_h_ig -= learning_rate * dWh_ig
        self.bobot_h_cc -= learning_rate * dWh_cc
        self.bobot_h_og -= learning_rate * dWh_og

        self.bias_fg -= learning_rate * db_fg
        self.bias_ig -= learning_rate * db_ig
        self.bias_cc -= learning_rate * db_cc
        self.bias_og -= learning_rate * db_og

        self.bobot_y -= learning_rate * dWy
        self.bias_y -= learning_rate * dby

    def save_model(self, filepath):
        """Save all weights and biases to a .npz file"""
        np.savez(
            filepath,
            # Input weights
            bobot_i_fg=self.bobot_i_fg,
            bobot_i_ig=self.bobot_i_ig,
            bobot_i_cc=self.bobot_i_cc,
            bobot_i_og=self.bobot_i_og,
            # Hidden weights
            bobot_h_fg=self.bobot_h_fg,
            bobot_h_ig=self.bobot_h_ig,
            bobot_h_cc=self.bobot_h_cc,
            bobot_h_og=self.bobot_h_og,
            # Biases
            bias_fg=self.bias_fg,
            bias_ig=self.bias_ig,
            bias_cc=self.bias_cc,
            bias_og=self.bias_og,
            # Output layer
            bobot_y=self.bobot_y,
            bias_y=self.bias_y,
            # Also save architecture info
            input_size=np.array([self.input_size]),
            hidden_size=np.array([self.hidden_size]),
            output_size=np.array([self.output_size])
        )
        print(f"Model saved to {filepath}.npz")

    def load_model(self, filepath):
        """Load weights and biases from a .npz file"""
        data = np.load(filepath)

        # Load weights & biases
        self.bobot_i_fg = data['bobot_i_fg']
        self.bobot_i_ig = data['bobot_i_ig']
        self.bobot_i_cc = data['bobot_i_cc']
        self.bobot_i_og = data['bobot_i_og']

        self.bobot_h_fg = data['bobot_h_fg']
        self.bobot_h_ig = data['bobot_h_ig']
        self.bobot_h_cc = data['bobot_h_cc']
        self.bobot_h_og = data['bobot_h_og']

        self.bias_fg = data['bias_fg']
        self.bias_ig = data['bias_ig']
        self.bias_cc = data['bias_cc']
        self.bias_og = data['bias_og']

        self.bobot_y = data['bobot_y']
        self.bias_y = data['bias_y']

        print(f"Model loaded from {filepath}")

input_size = 1
hidden_size = 32
output_size = 1

dataset = pd.read_csv("data/normalize/train.csv")
data = dataset["Total"].values
jumlah_hari = 2

input_train, target_train = [], []

for i in range(len(data) - jumlah_hari):
    input_train.append(np.array(data[i:i+jumlah_hari]).reshape(-1, 1))
    target_train.append(np.array(data[i+jumlah_hari]))

model = lstm(input_size, hidden_size, output_size)

steps = 50
learning_rate = 0.01
input_length = jumlah_hari

losses = []
for step in range(steps):
    total_loss = 0.0
    
    for i in range(len(input_train)):
        x_seq = input_train[i]      # shape (2, 1)
        y_true = target_train[i]    # scalar
        
        # Forward pass → returns prediction for next day (shape (1,1))
        y_pred = model.forward(x_seq)
        
        # Compute scalar loss (MSE)
        error = y_pred.item() - y_true
        loss = 0.5 * error ** 2
        total_loss += loss
        
        dy = np.array([[error]])
        
        dy_seq = [np.zeros_like(dy) for _ in range(input_length - 1)] + [dy]
        
        # Backward pass
        model.backward(dy_seq, learning_rate)
    
    avg_loss = total_loss / len(input_train)
    losses.append(avg_loss)
    
    if step % 5 == 0:
        print(f"Step {step}, Avg Loss: {avg_loss:.6f}")

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()

# model.save_model("m2")

# # Load trained weights
# model.load_model("m2.npz")  # or "models/sales_lstm_model.npz"

# # Prepare input: last 2 days of sales (shape: (2, 1))
# last_two_days = np.array([[0.25], [0.30]])  # example values

# # Predict!
# prediction = model.forward(last_two_days)
# print(f"Predicted next day sales: {prediction.item():.4f}")