import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import joblib

import streamlit as st
from io import BytesIO

class backpropagation:
    def __init__(self, learning_rate):
        self.bobot = np.array([np.random.randn(), np.random.randn()])
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
        total_error = []
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
                
                rata_rata = jumlah_error / len(input)
                total_error.append(rata_rata)

        return total_error, rata_rata
    
    def validasi(self, input, target):
        total_error = 0
        for i in range(len(input)):
            pred = self.prediksi(input[i])
            error = self.mse(pred, target[i])
            total_error += error

        rata_rata = total_error / len(input)

        return rata_rata, pred, error
    
    def test(self, input, scl):
        scaler = joblib.load(scl)
        output = np.zeros(len(input))
        
        total_error = 0
        for i in range(len(input)):
            pred = self.prediksi(input[i])
            output[i] = scaler.inverse_transform([[pred]])[0][0]
            error = self.mse(pred, target[i])
            total_error += error

        rata_rata = total_error / len(input)
        
        return output, scaler, rata_rata
    
    def save_model(self, file):
        np.savez(file, bobot = self.bobot, bias = self.bias)

    def load_model(self, file):
        parameter = np.load(file)
        self.bobot = parameter["bobot"]
        self.bias = parameter["bias"]

st.title("Backpropagation Prediksi Penjualan")

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

option = st.selectbox("Mode", ["=== Pilih Mode ===", "Training", "Validasi", "Testing"])

if option == "Training":
    learning_rate = st.number_input("Learning Rate", min_value=0.0, max_value=1.0, step=0.1)
    steps = st.number_input("Steps", min_value=0, step=100, format="%d")

    if learning_rate > 0 and steps > 0:
        if st.button("Mulai"):
            model = backpropagation(learning_rate)
            error, rata_rata = model.train(input, target, steps)
            
            if error is not None:
                fig, ax = plt.subplots()
                ax.plot(error)
                ax.set_title("Training Loss Setiap 100 Steps")
                ax.set_xlabel("Steps")
                ax.set_ylabel("Loss")

                st.pyplot(fig)

                st.write("Error Pertama: ", error[0])
                st.write("Error Terakhir: ", rata_rata)

                buffer = BytesIO()
                model.save_model(buffer)
                buffer.seek(0)
            
                st.download_button(
                    label="Download Model",
                    data=buffer,
                    file_name="model.npz",
                    mime="application/octet-stream"
                )

elif option == "Validasi":
    model = backpropagation(learning_rate=0.1)
    file = st.file_uploader("Unggah Model")

    if file is not None:
        model.load_model(file)
        rata_rata, pred, error = model.validasi(input, target)

        fig, ax1 = plt.subplots(figsize=(12,5))

        ax1.plot(target, label='Prediksi', linewidth=1)
        ax1.plot(pred, label='Target', linewidth=1)
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("Nilai")
        ax1.set_title("Perbandingan Prediksi dan Target")
        ax1.legend(loc="upper left")

        st.pyplot(fig)

        st.write("Error Rata-Rata: ", rata_rata)
elif option == "Testing":
    model = backpropagation(learning_rate=0.1)
    file = st.file_uploader("Unggah Model")
    scaler = st.file_uploader("Unggah Scaler")

    if jumlah_hari > 0:
        tanggal = tanggal[jumlah_hari:] 

    if file is not None and scaler is not None:
        model.load_model(file)
        prediksi, scaler, rata_rata = model.test(input, scaler)

        baris = []
        array_target = np.array(target)
        target_data = array_target.reshape(-1, 1)
        actual_target = scaler.inverse_transform(target_data)
        target_value = np.rint(actual_target).astype(int)

        for i in range(len(input)):
            row = {}
            row["Tanggal"] = tanggal[i]
            row["Prediksi"] = prediksi[i]
            row["Target"] = target_value[i]
            baris.append(row)

        output = pd.DataFrame(baris)
        st.write("Hasil Pengujian")
        st.dataframe(output)

        st.write("Error Rata-Rata: ", rata_rata)