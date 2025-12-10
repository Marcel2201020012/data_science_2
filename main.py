import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

import pandas as pd
import joblib

import streamlit as st
from io import BytesIO
import zipfile

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

class backpropagation_klasifikasi():
    def __init__(self, learning_rate=0.1, input=1):
        self.bobot = np.random.randn(input)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def cross_entropy(self, pred, target):
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        return - (target * np.log(pred) + (1 - target) * np.log(1 - pred))
    
    def prediksi(self, input):
        dot = np.dot(input, self.bobot) + self.bias
        aktivasi = self.sigmoid(dot)

        return aktivasi
    
    def gradient(self, input, target):
        dot = np.dot(input, self.bobot) + self.bias
        aktivasi = self.sigmoid(dot)

        turunan = aktivasi - target

        gradient_bias = turunan
        gradient_bobot = turunan * input

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
                    error = self.cross_entropy(pred, target[j])
                    jumlah_error += error
                
                error_per_steps.append(jumlah_error / len(input))

        return error_per_steps
    
    def validasi(self, input, target):
        output = []
        benar = 0
        total_error = 0

        for i in range(len(input)):
            pred_prob = self.prediksi(input[i])
            loss = self.cross_entropy(pred_prob, target[i])
            total_error += loss

            pred_label = 1 if pred_prob >= 0.5 else 0
            output.append(pred_label)

            if pred_label == target[i]:
                benar += 1

        akurasi = benar / len(input)
        rata_rata = total_error / len(input)

        return np.array(output), akurasi, rata_rata
    
    def save_model(self, file):
        np.savez(file, bobot = self.bobot, bias = self.bias)

    def load_model(self, file):
        parameter = np.load(file)
        self.bobot = parameter["bobot"]
        self.bias = parameter["bias"]

class lstm():
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
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

            #update cell state (long term)
            c = fg * prev_c + ig * cc

            #update hidden state (short term)
            h = og * np.tanh(c)

            #output
            y = self.bobot_y @ h + self.bias_y
            output.append(y)

            self.cache.append((input, prev_h, h, fg, ig, og, cc, prev_c, c))

        return output[-1]
    
    def backward(self, gradient_output):
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

        dy_final = gradient_output.reshape(-1, 1)
        dWy += dy_final @ self.cache[-1][2].T
        dby += dy_final

        dh_next = self.bobot_y.T @ dy_final
        dc_next = np.zeros((self.hidden_size, 1))

        #BPTT
        for i in reversed(range(len(self.cache))):
            x, h_prev, h, fg, ig, og, cc, c_prev, c = self.cache[i]

            dc = dh_next * og * self.dtanh(c) + dc_next

            dog = dh_next * np.tanh(c)
            dfg = dc * c_prev
            dig = dc * cc
            dcc = dc * ig

            og_input = self.bobot_i_og @ x + self.bobot_h_og @ h_prev + self.bias_og
            fg_input = self.bobot_i_fg @ x + self.bobot_h_fg @ h_prev + self.bias_fg
            ig_input = self.bobot_i_ig @ x + self.bobot_h_ig @ h_prev + self.bias_ig
            cc_input = self.bobot_i_cc @ x + self.bobot_h_cc @ h_prev + self.bias_cc

            dog_input = dog * self.dsigmoid(og_input)
            dfg_input = dfg * self.dsigmoid(fg_input)
            dig_input = dig * self.dsigmoid(ig_input)
            dcc_input = dcc * self.dtanh(cc_input)

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

            dh_next = (self.bobot_h_fg.T @ dfg_input +
                       self.bobot_h_ig.T @ dig_input +
                       self.bobot_h_cc.T @ dcc_input +
                       self.bobot_h_og.T @ dog_input)

            dc_next = dc * fg

        #Update parameters
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
        np.savez(
            filepath,

            bobot_i_fg=self.bobot_i_fg,
            bobot_i_ig=self.bobot_i_ig,
            bobot_i_cc=self.bobot_i_cc,
            bobot_i_og=self.bobot_i_og,

            bobot_h_fg=self.bobot_h_fg,
            bobot_h_ig=self.bobot_h_ig,
            bobot_h_cc=self.bobot_h_cc,
            bobot_h_og=self.bobot_h_og,

            bias_fg=self.bias_fg,
            bias_ig=self.bias_ig,
            bias_cc=self.bias_cc,
            bias_og=self.bias_og,

            # Output layer
            bobot_y=self.bobot_y,
            bias_y=self.bias_y,

            input_size=np.array([self.input_size]),
            hidden_size=np.array([self.hidden_size]),
            output_size=np.array([self.output_size])
        )

    def load_model(self, filepath):
        data = np.load(filepath)

        self.bobot_i_fg = data["bobot_i_fg"]
        self.bobot_i_ig = data["bobot_i_ig"]
        self.bobot_i_cc = data["bobot_i_cc"]
        self.bobot_i_og = data["bobot_i_og"]

        self.bobot_h_fg = data["bobot_h_fg"]
        self.bobot_h_ig = data["bobot_h_ig"]
        self.bobot_h_cc = data["bobot_h_cc"]
        self.bobot_h_og = data["bobot_h_og"]

        self.bias_fg = data["bias_fg"]
        self.bias_ig = data["bias_ig"]
        self.bias_cc = data["bias_cc"]
        self.bias_og = data["bias_og"]

        self.bobot_y = data["bobot_y"]
        self.bias_y = data["bias_y"]

class lstm_klasifikasi():
    def __init__(self, vocab_size, dimensi_embedding, hidden_size, output_size, learning_rate):
        self.vocab_size = vocab_size
        self.dimensi_embedding = dimensi_embedding
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        #layer embedding [embed_dim, vocab_size]
        self.embedding = np.random.randn(dimensi_embedding, vocab_size) * np.sqrt(1.0 / dimensi_embedding)

        #bobot input
        self.bobot_i_fg = np.random.randn(hidden_size, dimensi_embedding)
        self.bobot_i_ig = np.random.randn(hidden_size, dimensi_embedding)
        self.bobot_i_cc = np.random.randn(hidden_size, dimensi_embedding)
        self.bobot_i_og = np.random.randn(hidden_size, dimensi_embedding)

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

        for i in range(len(input_sequence)):
            word_id = int(input_sequence[i])  # e.g., 5

            #embedding lookup
            if word_id >= self.vocab_size:
                word_id = 1  # <UNK>
            input = self.embedding[:, word_id].reshape(-1, 1)  #(embed_dim, 1)

            fg = self.sigmoid(self.bobot_i_fg @ input + self.bobot_h_fg @ prev_h + self.bias_fg)
            ig = self.sigmoid(self.bobot_i_ig @ input + self.bobot_h_ig @ prev_h + self.bias_ig)
            cc = np.tanh(self.bobot_i_cc @ input + self.bobot_h_cc @ prev_h + self.bias_cc)
            og = self.sigmoid(self.bobot_i_og @ input + self.bobot_h_og @ prev_h + self.bias_og)

            #update cell state (long term)
            c = fg * prev_c + ig * cc

            #update hidden state (short term)
            h = og * np.tanh(c)

            self.cache.append((word_id, input, prev_h, h, fg, ig, og, cc, prev_c, c))
            prev_h, prev_c = h, c

        #output layer
        logits = self.bobot_y @ h + self.bias_y
        prob = self.sigmoid(logits)  # 0 - 1

        return prob
    
    def backward(self, gradient_output):
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

        d_embedding = np.zeros_like(self.embedding)

        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))

        #BPTT
        for i in reversed(range(len(gradient_output))):
            word_id, x, h_prev, h, fg, ig, og, cc, c_prev, c = self.cache[i]

            dy = gradient_output[i].reshape(-1, 1)
            dWy += dy @ h.T
            dby += dy

            dh = self.bobot_y.T @ dy + dh_next
            dc = dh * og * self.dtanh(c) + dc_next

            dog = dh * np.tanh(c)
            dfg = dc * c_prev
            dig = dc * cc
            dcc = dc * ig

            dog_input = dog * self.dsigmoid(self.bobot_i_og @ x + self.bobot_h_og @ h_prev + self.bias_og)
            dfg_input = dfg * self.dsigmoid(self.bobot_i_fg @ x + self.bobot_h_fg @ h_prev + self.bias_fg)
            dig_input = dig * self.dsigmoid(self.bobot_i_ig @ x + self.bobot_h_ig @ h_prev + self.bias_ig)
            dcc_input = dcc * self.dtanh(self.bobot_i_cc @ x + self.bobot_h_cc @ h_prev + self.bias_cc)

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

            dx = (self.bobot_i_fg.T @ dfg_input +
                  self.bobot_i_ig.T @ dig_input +
                  self.bobot_i_cc.T @ dcc_input +
                  self.bobot_i_og.T @ dog_input)
            
            d_embedding[:, word_id] += dx.flatten()  # dx (embed_dim, 1)

            dh_next = (self.bobot_h_fg.T @ dfg_input +
                       self.bobot_h_ig.T @ dig_input +
                       self.bobot_h_cc.T @ dcc_input +
                       self.bobot_h_og.T @ dog_input)

            dc_next = dc * fg

        #Update parameters
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

        self.embedding -= self.learning_rate * d_embedding

        self.bobot_y -= self.learning_rate * dWy
        self.bias_y -= self.learning_rate * dby

    def save_model(self, filepath):
        np.savez(
            filepath,

            bobot_i_fg=self.bobot_i_fg,
            bobot_i_ig=self.bobot_i_ig,
            bobot_i_cc=self.bobot_i_cc,
            bobot_i_og=self.bobot_i_og,

            bobot_h_fg=self.bobot_h_fg,
            bobot_h_ig=self.bobot_h_ig,
            bobot_h_cc=self.bobot_h_cc,
            bobot_h_og=self.bobot_h_og,

            bias_fg=self.bias_fg,
            bias_ig=self.bias_ig,
            bias_cc=self.bias_cc,
            bias_og=self.bias_og,

            embedding=self.embedding,

            # Output layer
            bobot_y=self.bobot_y,
            bias_y=self.bias_y,

            vocab_size=np.array([self.vocab_size]),
            dimensi_embedding=np.array([self.dimensi_embedding]),
            hidden_size=np.array([self.hidden_size]),
            output_size=np.array([self.output_size])
        )

    def load_model(self, filepath):
        data = np.load(filepath)

        self.bobot_i_fg = data["bobot_i_fg"]
        self.bobot_i_ig = data["bobot_i_ig"]
        self.bobot_i_cc = data["bobot_i_cc"]
        self.bobot_i_og = data["bobot_i_og"]

        self.bobot_h_fg = data["bobot_h_fg"]
        self.bobot_h_ig = data["bobot_h_ig"]
        self.bobot_h_cc = data["bobot_h_cc"]
        self.bobot_h_og = data["bobot_h_og"]

        self.bias_fg = data["bias_fg"]
        self.bias_ig = data["bias_ig"]
        self.bias_cc = data["bias_cc"]
        self.bias_og = data["bias_og"]

        self.embedding = data["embedding"]

        self.bobot_y = data["bobot_y"]
        self.bias_y = data["bias_y"]

studi_kasus = st.selectbox("Studi Kasus", ["=== Pilih Studi Kasus ===", "Prediksi", "Klasifikasi"])

if studi_kasus == "Prediksi":
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

                    ax.plot(pred, label="Prediksi", linewidth=1, color="blue")
                    ax.plot(target_value, label="Target", linewidth=1, color="red")

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

                            for i in range(len(input_train)):
                                input_seq = input_train[i]
                                target_train = target[i]

                                pred = model.forward(input_seq)

                                #mse
                                error = pred.item() - target_train
                                loss = 0.5 * error ** 2
                                total_loss += loss

                                dy = np.array([[error]])

                                model.backward(dy)

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
                model = lstm(input_size=1,
                            hidden_size=32,
                            output_size=1,
                            learning_rate=0.1)
                file = st.file_uploader("Unggah Model")
                scaler = st.file_uploader("Unggah Scaler")

                if jumlah_hari > 0 and "data" in locals() and len(data) > jumlah_hari:
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

                        #mse
                        error = pred.item() - target[i]
                        loss = 0.5 * error ** 2
                        total_loss += loss

                    total_loss /= len(input)

                    pred_array = np.array(predictions_normalized).reshape(-1, 1)
                    target_array = np.array(target).reshape(-1, 1)

                    pred_denorm = scl.inverse_transform(pred_array).flatten()
                    target_denorm = scl.inverse_transform(target_array).flatten()

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

                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(pred_denorm, label="Prediksi", color="blue", linewidth=1)
                    ax.plot(target_denorm, label="Target", color="red", linewidth=1)
                    ax.set_xlabel("Sample")
                    ax.set_ylabel("Nilai")
                    ax.set_title("Perbandingan Prediksi dan Target")
                    ax.legend(loc="upper left")
                    st.pyplot(fig)

                    st.write(f"Error Rata-Rata: ", total_loss)
elif studi_kasus == "Klasifikasi":
    st.title("Klasifikasi Sentimen")

    file = st.file_uploader("Unggah Dataset Dalam Format CSV")
    dataset = None
    if file is not None:
        dataset = pd.read_csv(file)

        if "show_preview" not in st.session_state:
            st.session_state.show_preview = False

        def preview():
            st.session_state.show_preview = not st.session_state.show_preview

        label_preview = "Hide" if st.session_state.show_preview else "Preview"

        if st.button(label_preview, on_click=preview):
            pass

        if st.session_state.show_preview:
            baris = []

            for label, text in dataset[["Sentiment", "Instagram Comment Text"]].values:
                row = {}
                row["Teks"] = text
                row["Sentimen"] = label
                baris.append(row)

            preview = pd.DataFrame(baris)
            st.write("Preview Data")
            st.dataframe(preview)

    methode = st.selectbox("Metode", ["=== Pilih Metode ===", "Backpropagation", "LSTM"]) 

    if methode != "=== Pilih Metode ===":
        option = st.selectbox("Mode", ["=== Pilih Mode ===", "Training", "Validasi/Testing"])

        if methode == "Backpropagation":
            all_words = []
            labels = []
            texts = []

            for label, text in dataset[["Sentiment", "Instagram Comment Text"]].values:
                words = text.lower().split()
                all_words.extend(words)
                texts.append(words)
                labels.append(label)

            vocab = {}
            for word in sorted(set(all_words)):
                vocab[word] = len(vocab)

            vocab_size = len(vocab)

            #konversi teks ke binary bag of words
            def text_to_bow(words, vocab):
                vec = np.zeros(len(vocab))
                for word in words:
                    if word in vocab:
                        vec[vocab[word]] = 1
                return vec

            input_vectors = []
            for words in texts:
                vec = text_to_bow(words, vocab)
                input_vectors.append(vec)

            input = np.array(input_vectors)
            target = np.array(labels, dtype=float)

            if option == "Training":
                learning_rate = st.number_input("Learning Rate", min_value=0.0, max_value=1.0, step=0.1)
                steps = st.number_input("Steps", min_value=0, step=100, format="%d")

                model = backpropagation_klasifikasi(learning_rate, vocab_size)

                if learning_rate > 0 and steps > 0:
                    if st.button("Mulai"):
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
                            vocab_buffer = BytesIO()

                            joblib.dump(vocab, vocab_buffer)
                            vocab_buffer.seek(0)

                            model.save_model(buffer)
                            buffer.seek(0)

                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w") as z:
                                z.writestr("vocab.pkl", vocab_buffer.getvalue())
                                z.writestr("model.npz", buffer.getvalue())

                            zip_buffer.seek(0)

                            st.download_button(
                                label="Download Model",
                                data=zip_buffer,
                                file_name="model.zip",
                                mime="application/zip"
                                )
            elif option == "Validasi/Testing":
                file = st.file_uploader("Unggah Model")
                vocab_file = st.file_uploader("Unggah vocab")

                if file is not None and vocab_file is not None:
                    vocab = joblib.load(vocab_file)
                    vocab_size = len(vocab)

                    #konversi teks ke binary bag of words
                    def text_to_bow(words, vocab):
                        vec = np.zeros(len(vocab))
                        for word in words:
                            if word in vocab:
                                vec[vocab[word]] = 1
                        return vec

                    input_vectors = []
                    for words in texts:
                        vec = text_to_bow(words, vocab)
                        input_vectors.append(vec)

                    input = np.array(input_vectors)
                    target = np.array(labels, dtype=float)

                    model = backpropagation_klasifikasi(input=vocab_size)
                    model.load_model(file)
                    output, akurasi, rata_rata = model.validasi(input, target)

                    baris = []

                    for i in range(len(input)):
                        row = {}
                        row["Prediksi"] = output[i]
                        row["Target"] = target[i]
                        baris.append(row)

                    output_df = pd.DataFrame(baris)
                    st.write("Hasil Pengujian")
                    st.dataframe(output_df)

                    st.write(f"Akurasi: {akurasi:.2%}")

                    cm = confusion_matrix(target, output)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Prediksi")
                    ax.set_ylabel("Aktual")
                    st.pyplot(fig)

                    st.write(f"Error Rata-Rata: ", rata_rata)
        elif methode == "LSTM" and dataset:
            all_words = []
            labels = []
            texts = []

            for label, text in dataset[["Sentiment", "Instagram Comment Text"]].values:
                words = text.lower().split()
                all_words.extend(words)
                texts.append(words)
                labels.append(label)

            vocab = {"<PAD>": 0, "<UNK>": 1}
            for i in sorted(set(all_words)):
                vocab[i] = len(vocab)

            #maxlen = sequence length
            def text_to_seq(words, maxlen=20):
                seq = [vocab.get(w, vocab["<UNK>"]) for w in words]

                if len(seq) < maxlen: #tambah padding ke text
                    seq = [vocab["<PAD>"]] * (maxlen - len(seq)) + seq
                else:
                    seq = seq[-maxlen:]  #simpan text maxlen terakhir
                return seq

            input = np.array([text_to_seq(words) for words in texts])
            target = np.array(labels).reshape(-1, 1)  # shape: (N, 1)

            if option == "Training":
                learning_rate = st.number_input("Learning Rate", min_value=0.0, max_value=1.0, step=0.01)
                steps = st.number_input("Steps", min_value=0, step=100, format="%d")

                if learning_rate > 0 and steps > 0:
                    if st.button("Mulai"):
                        vocab_size = len(vocab)
                        dimensi_embedding = 32
                        hidden_size = 32
                        output_size = 1

                        model = lstm_klasifikasi(vocab_size, dimensi_embedding, hidden_size, output_size, learning_rate)

                        losses = []
                        for step in range(steps):
                            total_loss = 0
                            for i in range(len(input)):
                                input_seq = input[i]
                                target_train = target[i]

                                # Forward
                                pred = model.forward(input_seq)
                                p = np.clip(pred.item(), 1e-15, 1 - 1e-15)
                                loss = -(target_train * np.log(p) + (1 - target_train) * np.log(1 - p))
                                total_loss += loss

                                # Gradient for BCE
                                dy = np.array([[p - target_train]])
                                dy_seq = [np.zeros_like(dy) for _ in range(len(input_seq) - 1)] + [dy]
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
                            vocab_buffer = BytesIO()

                            joblib.dump(vocab, vocab_buffer)
                            vocab_buffer.seek(0)

                            model.save_model(buffer)
                            buffer.seek(0)

                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w") as z:
                                z.writestr("vocab.pkl", vocab_buffer.getvalue())
                                z.writestr("model.npz", buffer.getvalue())

                            zip_buffer.seek(0)

                            st.download_button(
                                label="Download Model",
                                data=zip_buffer,
                                file_name="model.zip",
                                mime="application/zip"
                                )
            elif option == "Validasi/Testing":
                file = st.file_uploader("Unggah Model")
                vocab_file = st.file_uploader("Unggah vocab")

                if file is not None and vocab_file is not None:
                    vocab = joblib.load(vocab_file)

                    vocab_size = len(vocab)
                    dimensi_embedding = 32
                    hidden_size = 32
                    output_size = 1
                    learning_rate = 0.1   

                    model = lstm_klasifikasi(vocab_size, dimensi_embedding, hidden_size, output_size, learning_rate)
                    model.load_model(file)

                    output = []

                    total_loss = 0
                    for i in range(len(input)):
                        pred = model.forward(input[i]).item()
                        target_train = target[i].item()

                        p = np.clip(pred, 1e-15, 1 - 1e-15)
                        loss = -(target_train * np.log(p) + (1 - target_train) * np.log(1 - p))
                        total_loss += loss

                        if pred > 0.5:
                            pred = 1
                        else:
                            pred = 0

                        output.append({
                            "Prediksi": pred,
                            "Target": target_train
                        })

                    df_results = pd.DataFrame(output)
                    st.write("Hasil Pengujian")
                    st.dataframe(df_results)

                    pred_list = [o["Prediksi"] for o in output]
                    target_list = [o["Target"] for o in output]

                    akurasi = accuracy_score(target_list, pred_list)
                    st.write(f"Akurasi: {akurasi:.2%}")

                    cm = confusion_matrix(target_list, pred_list)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Prediksi")
                    ax.set_ylabel("Aktual")
                    st.pyplot(fig)

                    st.write(f"Error Rata-Rata: ", total_loss / len(input))