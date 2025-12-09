import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import joblib

import streamlit as st
from io import BytesIO

class backpropagation:
    def __init__(self, learning_rate=0.1, input=1):
        self.bobot = np.array(np.random.randn(input))
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
    
    def sigmoid(self, dot_product):
        return 1 / (1 + np.exp(-dot_product))
    
    def mse(self, pred, target):
        return (pred - target)**2
    
    def prediksi(self, input):
        dot = np.dot(input, self.bobot) + self.bias
        aktivasi = self.sigmoid(dot)

        return aktivasi
    
    def gradient(self, input, target):
        dot = np.dot(input, self.bobot) + self.bias
        aktivasi = self.sigmoid(dot)

        turunan_mse = 2 * (aktivasi - target)
        turunan_sigmoid = aktivasi * (1 - aktivasi)
        turunan_bias = 1
        turunan_bobot = input

        gradient_bias = (turunan_mse * turunan_sigmoid * turunan_bias)
        gradient_bobot = (turunan_mse * turunan_sigmoid * turunan_bobot)

        return gradient_bias, gradient_bobot
    
    def update_parameter(self, gradient_bias, gradient_bobot):
        self.bias -= gradient_bias * self.learning_rate
        self.bobot -= gradient_bobot * self.learning_rate

    def train(self, input, target, steps):
        error_per_steps = []
        for i in range(steps):
            random_index = np.random.randint(len(input))

            input_index = input[random_index]
            target_index = target[random_index]

            gradient_bias, gradient_bobot = self.gradient(input_index, target_index)
            self.update_parameter(gradient_bias, gradient_bobot)

            if i % 100 == 0:
                jumlah_error = 0
                for j in range(len(input)):
                    pred = self.prediksi(input[j])
                    error = self.mse(pred, target[j])
                    jumlah_error += error
                
                error_per_steps.append(jumlah_error / len(input))

        return error_per_steps
    
    def validasi(self, input, target, scl):
        scaler = joblib.load(scl)
        output = np.zeros(len(input))

        total_error = 0
        for i in range(len(input)):
            pred = self.prediksi(input[i])
            error = self.mse(pred, target[i])
            total_error += error
            output[i] = scaler.inverse_transform([[pred]])[0][0]

        rata_rata = total_error / len(input)

        return output, scaler, rata_rata
    
    def save_model(self, file):
        np.savez(file, bobot = self.bobot, bias = self.bias)

    def load_model(self, file):
        parameter = np.load(file)
        self.bobot = parameter["bobot"]
        self.bias = parameter["bias"]

class lstm():
    def __init__(self, input_size=1, hidden_size=32, output_size=1, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

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
    
    def backward(self, gradient_output):
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
        self.bobot_i_fg -= self.learning_rate * dWi_fg
        self.bobot_i_ig -= self.learning_rate * dWi_ig
        self.bobot_i_cc -= self.learning_rate * dWi_cc
        self.bobot_i_og -= self.learning_rate * dWi_og

        self.bobot_h_fg -= self.learning_rate * dWh_fg
        self.bobot_h_ig -= self.learning_rate * dWh_ig
        self.bobot_h_cc -= self.learning_rate * dWh_cc
        self.bobot_h_og -= self.learning_rate * dWh_og

        self.bias_fg -= self.learning_rate * db_fg
        self.bias_ig -= self.learning_rate * db_ig
        self.bias_cc -= self.learning_rate * db_cc
        self.bias_og -= self.learning_rate * db_og

        self.bobot_y -= self.learning_rate * dWy
        self.bias_y -= self.learning_rate * dby

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

st.title("Prediksi Penjualan")

file = st.file_uploader("Unggah Dataset Dalam Format CSV")
jumlah_hari = st.number_input("Masukkan Jumlah Hari", min_value=0, step=1, format="%d")

if file is not None and jumlah_hari > 0:
    dataset = pd.read_csv(file)
    data = dataset["Total"].values
    tanggal = dataset["Tanggal"].values

    input, target = [], []

    for i in range(len(data) - jumlah_hari):
        input.append(np.array(data[i:i+jumlah_hari]))
        target.append(data[i+jumlah_hari])

    if "show_preview" not in st.session_state:
        st.session_state.show_preview = False

    def preview():
        st.session_state.show_preview = not st.session_state.show_preview

    label_preview = "Hide" if st.session_state.show_preview else "Preview"

    if st.button(label_preview, on_click=preview):
        pass

    if st.session_state.show_preview:
        baris = []

        for i in range(len(input)):
            row = {}
            row["Input"] = str(input[i])
            row["Target"] = target[i]
            baris.append(row)

        preview = pd.DataFrame(baris)
        st.write("Preview Data")
        st.dataframe(preview)

methode = st.selectbox("Metode", ["=== Pilih Metode ===", "Backpropagation", "LSTM"]) 

if methode != "=== Pilih Metode ===":
    option = st.selectbox("Mode", ["=== Pilih Mode ===", "Training", "Validasi/Testing"])

    if methode == "Backpropagation":
        if option == "Training":
            learning_rate = st.number_input("Learning Rate", min_value=0.0, max_value=1.0, step=0.1)
            steps = st.number_input("Steps", min_value=0, step=100, format="%d")
    
            if learning_rate > 0 and steps > 0:
                if st.button("Mulai"):
                    model = backpropagation(learning_rate, jumlah_hari)
                    error = model.train(input, target, steps)
    
                    if error is not None:
                        fig, ax = plt.subplots()
                        ax.plot(error)
                        ax.set_title("Training Loss Setiap 100 Steps")
                        ax.set_xlabel("Steps")
                        ax.set_ylabel("Loss")
    
                        st.pyplot(fig)
    
                        st.write("Error Pertama: ", error[0])
                        st.write("Error Terakhir: ", error[-1])
    
                        buffer = BytesIO()
                        model.save_model(buffer)
                        buffer.seek(0)
    
                        st.download_button(
                            label="Download Model",
                            data=buffer,
                            file_name="model.npz",
                            mime="application/octet-stream"
                        )
    
        elif option == "Validasi/Testing":
            model = backpropagation()
            file = st.file_uploader("Unggah Model")
            scaler = st.file_uploader("Unggah Scaler")
    
            if jumlah_hari > 0:
                tanggal = tanggal[jumlah_hari:] 
    
            if file is not None and scaler is not None:
                model.load_model(file)
                pred, scaler, rata_rata = model.validasi(input, target, scaler)
    
                baris = []
                array_target = np.array(target)
                target_data = array_target.reshape(-1, 1)
                actual_target = scaler.inverse_transform(target_data)
                target_value = np.rint(actual_target).astype(int)
    
                for i in range(len(input)):
                    row = {}
                    row["Tanggal"] = tanggal[i]
                    row["Prediksi"] = pred[i]
                    row["Target"] = target_value[i]
                    baris.append(row)
    
                output = pd.DataFrame(baris)
                st.write("Hasil Pengujian")
                st.dataframe(output)
    
                fig, ax = plt.subplots(figsize=(12,5))
    
                ax.plot(pred, label='Prediksi', linewidth=1, color='blue')
                ax.plot(target_value, label='Target', linewidth=1, color='red')
    
                ax.set_xlabel("Sample")
                ax.set_ylabel("Nilai")
                ax.set_title("Perbandingan Prediksi dan Target")
                ax.legend(loc="upper left")
    
                st.pyplot(fig)
    
                st.write("Error Rata-Rata: ", rata_rata)
    elif methode == "LSTM":
        if option == "Training":
            learning_rate = st.number_input("Learning Rate", min_value=0.0, max_value=1.0, step=0.01)
            steps = st.number_input("Steps", min_value=0, step=100, format="%d")
    
            if learning_rate > 0 and steps > 0:
                if st.button("Mulai"):
                    input_train = [np.array(data[i:i+jumlah_hari]).reshape(-1, 1) for i in range(len(data) - jumlah_hari)]
                    target_train = data[jumlah_hari:]


                    model = lstm(
                        input_size=1,
                        hidden_size=32,
                        output_size=1,
                        learning_rate=learning_rate
                        )
    
                    losses = []
                    for step in range(steps):
                        total_loss = 0.0
    
                        for i in range(len(input)):
                            x_seq = input[i]
                            y_true = target[i]
    
                            y_pred = model.forward(x_seq)
    
                            error = y_pred.item() - y_true
                            loss = 0.5 * error ** 2
                            total_loss += loss
    
                            dy = np.array([[error]])
    
                            dy_seq = [np.zeros_like(dy) for _ in range(jumlah_hari - 1)] + [dy]
    
                            model.backward(dy_seq)
    
                        avg_loss = total_loss / len(input)
                        losses.append(avg_loss)
    
                    if losses:
                        fig, ax = plt.subplots()
                        ax.plot(losses)
                        ax.set_title("Training Loss")
                        ax.set_xlabel("Steps")
                        ax.set_ylabel("Loss")
                        st.pyplot(fig)
                        st.write("Error Pertama: ", losses[0])
                        st.write("Error Terakhir: ", losses[-1])

                        buffer = BytesIO()
                        model.save_model(buffer)
                        buffer.seek(0)

                        st.download_button(
                            label="Download Model",
                            data=buffer,
                            file_name="model.npz",
                            mime="application/octet-stream"
                        )
        elif option == "Validasi/Testing":
            model = lstm()
            file = st.file_uploader("Unggah Model")
            scaler = st.file_uploader("Unggah Scaler")
    
            if jumlah_hari > 0 and 'data' in locals() and len(data) > jumlah_hari:
                input = [np.array(data[i:i+jumlah_hari]).reshape(-1, 1) for i in range(len(data) - jumlah_hari)]
                target = data[jumlah_hari:]
                tanggal = tanggal[jumlah_hari:]
    
            if file is not None and scaler is not None:
                model.load_model(file)
                scl = joblib.load(scaler)
    
                predictions_normalized = []
                total_loss = 0.0

                for i in range(len(input)):
                    pred = model.forward(input[i])
                    predictions_normalized.append(pred.item())

                    error = pred.item() - target[i]
                    loss = 0.5 * error ** 2
                    total_loss += loss

                total_loss /= len(input)
                
                # === Denormalize predictions and targets ===
                pred_array = np.array(predictions_normalized).reshape(-1, 1)
                target_array = np.array(target).reshape(-1, 1)
                
                pred_denorm = scl.inverse_transform(pred_array).flatten()
                target_denorm = scl.inverse_transform(target_array).flatten()
                
                # Round to integers (if sales are whole numbers)
                target_rounded = np.rint(target_denorm).astype(int)
    
                results = []
                for i in range(len(pred_denorm)):
                    results.append({
                        "Tanggal": tanggal[i],
                        "Prediksi": pred_denorm[i],
                        "Target": target_rounded[i]
                    })

                df_results = pd.DataFrame(results)
                st.write("Hasil Pengujian")
                st.dataframe(df_results)

                # === Plot ===
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(pred_denorm, label='Prediksi', color='blue', linewidth=1)
                ax.plot(target_denorm, label='Target', color='red', linewidth=1)
                ax.set_xlabel("Sample")
                ax.set_ylabel("Nilai")
                ax.set_title("Perbandingan Prediksi dan Target")
                ax.legend(loc="upper left")
                st.pyplot(fig)

                # === Metrics ===
                st.write(f"Error Rata-Rata: ", total_loss)