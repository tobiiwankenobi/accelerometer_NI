# analyssis_iso.py – Kompletny raport ISO 2631-1 z FFT (0–100 Hz), VDV, CF i PDF z opisem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import scipy.signal as signal
import os
from pathlib import Path
from tkinter import Tk, filedialog, messagebox
import tkinter as tk
from tkinter import ttk
from datetime import datetime

class ISO2631Analyzer:
    def __init__(self, csv_path, sensitivity=1.0, sampling_rate=10000):
        self.csv_path = Path(csv_path)
        self.sensitivity = sensitivity
        self.fs = sampling_rate
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.csv_path, skiprows=2)
        self.time = self.df.iloc[:, 0].values
        self.signals = self.df.iloc[:, 1:].values * self.sensitivity

        dt = np.diff(self.time)
        if len(dt) == 0 or np.any(dt <= 0) or np.std(dt) > 0.01:
            print("⚠️ Time vector invalid — regenerating using sampling rate.")
            self.time = np.arange(len(self.signals)) / self.fs

    def apply_weighting(self, signal, axis):
        if axis == 'z':
            return self.filter_weighted(signal, 'wk')
        elif axis in ['x', 'y']:
            return self.filter_weighted(signal, 'wd')
        else:
            raise ValueError("Unknown axis")

    def filter_weighted(self, data, filter_type):
        freqs, weights = self.get_weighting(filter_type)
        freq_gain_pairs = sorted(set(zip(freqs, weights)))
        freqs = np.array([f for f, _ in freq_gain_pairs])
        gains = np.array([g for _, g in freq_gain_pairs]) / 1000.0

        if not np.all(np.diff(freqs) > 0):
            raise ValueError("Frequencies must be strictly increasing for filter design")

        freqs = np.concatenate(([0], freqs, [self.fs / 2]))
        gains = np.concatenate(([0], gains, [0]))

        taps = signal.firwin2(numtaps=501, freq=freqs, gain=gains, fs=self.fs)
        return signal.filtfilt(taps, [1.0], data)

    def get_weighting(self, filter_type):
        weightings = {
            'wd': {
                0.4: 0, 0.63: 9.2, 1: 16.2, 1.6: 23.1, 2.5: 29.6,
                4: 34.6, 6.3: 38.5, 10: 40.7, 16: 41.3, 25: 40.1,
                40: 36.2, 63: 29.3, 100: 19.8
            },
            'wk': {
                0.4: 0, 0.63: 6.5, 1: 13.4, 1.6: 19.8, 2.5: 26.0,
                4: 31.0, 6.3: 34.7, 10: 36.5, 16: 36.6, 25: 33.9,
                40: 28.2, 63: 19.3, 100: 7.4
            }
        }
        weights = weightings[filter_type]
        return np.array(list(weights.keys())), np.array(list(weights.values()))

    def calculate_rms(self, signal):
        return np.sqrt(np.mean(signal**2))

    def calculate_vdv(self, signal):
        return (np.sum(signal**4) * (1 / self.fs))**0.25

    def crest_factor(self, signal):
        return np.max(np.abs(signal)) / self.calculate_rms(signal)

    def fft_limited(self, sig_data, fmax=80):
        f, Pxx = signal.welch(sig_data, fs=self.fs, nperseg=2048)
        mask = f <= fmax
        return f[mask], Pxx[mask]

    def analyze(self):
        base_dir = Path(r"D:/1UNIVERSITY/10th semester/master thesis/code/python_reading_acc/csv_analiza")
        car_folder = self.csv_path.parents[1].name + "_analiza"
        output_dir = base_dir / car_folder / self.csv_path.parent.name

        # Tworzenie folderu pomiarowego z timestampem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        measurement_folder = output_dir / f"measurement_{self.csv_path.stem}_{timestamp}"
        measurement_folder.mkdir(parents=True, exist_ok=True)

        pdf_path = measurement_folder / f"{self.csv_path.stem}_full_report.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

        rms_results = {}
        vector_sum_sq = 0
        axes = ['x', 'y', 'z', 'z']
        gains = {'x': 1.4, 'y': 1.4, 'z': 1.0}
        description_lines = [f"Raport ISO 2631-1\nData: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                            f"Plik: {self.csv_path.name}\n"]

        for i, axis in enumerate(axes):
            try:
                raw = self.signals[:, i]
                filtered = self.apply_weighting(raw, axis)
                rms = self.calculate_rms(filtered)
                vdv = self.calculate_vdv(filtered)
                crest = self.crest_factor(filtered)
                f_fft, P_fft = self.fft_limited(filtered)

                rms_results[f'channel_{axis}_{i+1}_rms'] = rms
                rms_results[f'channel_{axis}_{i+1}_vdv'] = vdv
                rms_results[f'channel_{axis}_{i+1}_crest'] = crest

                if axis in gains:
                    vector_sum_sq += (gains[axis] * rms)**2

                # Tekstowy opis kanału
                description_lines.append(f"Kanał {i+1} (oś {axis}):")
                description_lines.append(f"  RMS = {rms:.4f} m/s²")
                description_lines.append(f"  VDV = {vdv:.4f} m/s⁴⁰.25")
                description_lines.append(f"  Crest Factor = {crest:.2f} → {'Uwaga na piki!' if crest > 9 else 'OK'}\n")

                # Wykres surowy sygnał
                fig_raw, ax_raw = plt.subplots()
                ax_raw.plot(self.time, raw)
                ax_raw.set_title(f"Kanał {i+1} — surowy sygnał")
                ax_raw.set_xlabel("Czas [s]")
                ax_raw.set_ylabel("Przyspieszenie [g]")
                ax_raw.grid()
                pdf.savefig(fig_raw)
                fig_raw.savefig(measurement_folder / f"channel_{i+1}_raw.svg", format='svg')
                plt.close(fig_raw)

                # Wykres po nałożeniu wag
                fig_filtered, ax_filtered = plt.subplots()
                ax_filtered.plot(self.time, filtered)
                ax_filtered.set_title(f"Kanał {i+1} — po nałożeniu wag ({axis.upper()})")
                ax_filtered.set_xlabel("Czas [s]")
                ax_filtered.set_ylabel("Przyspieszenie [g]")
                ax_filtered.grid()
                pdf.savefig(fig_filtered)
                fig_filtered.savefig(measurement_folder / f"channel_{i+1}_weighted.svg", format='svg')
                plt.close(fig_filtered)

                # Wykres FFT (0–80 Hz)
                fig_fft, ax_fft = plt.subplots()
                ax_fft.plot(f_fft, P_fft)
                ax_fft.set_title(f"Kanał {i+1} — widmo (0–80 Hz)")
                ax_fft.set_xlabel("Częstotliwość [Hz]")
                ax_fft.set_ylabel("Amplituda")
                ax_fft.grid()
                pdf.savefig(fig_fft)
                fig_fft.savefig(measurement_folder / f"channel_{i+1}_fft.svg", format='svg')
                plt.close(fig_fft)

            except Exception as e:
                description_lines.append(f"❌ Błąd przetwarzania kanału {i+1}: {e}\n")

        vector_sum = np.sqrt(vector_sum_sq) if vector_sum_sq > 0 else 0
        rms_results['vector_sum_rms'] = vector_sum
        description_lines.append(f"\nSumaryczna ocena (Vector RMS): {vector_sum:.4f} m/s²")

        if vector_sum < 0.315:
            desc = "Komfort dobry (poniżej 0.315 m/s²)"
        elif vector_sum < 0.63:
            desc = "🟡 Komfort akceptowalny"
        else:
            desc = "🔴 Uciążliwe drgania — może wpływać na zdrowie"
        description_lines.append(f"Ocena komfortu: {desc}\n")

        # Zapis CSV z wynikami
        results_path = measurement_folder / f"{self.csv_path.stem}_results.csv"
        pd.DataFrame([rms_results]).to_csv(results_path, index=False)

        # Tekstowy opis w PDF
        fig_text = plt.figure(figsize=(8.3, 11.7))
        fig_text.clf()
        txt = "\n".join(description_lines)
        fig_text.text(0.05, 0.95, txt, va='top', ha='left', fontsize=10, wrap=True)
        pdf.savefig(fig_text)
        plt.close(fig_text)

        pdf.close()
        return results_path


# GUI
class AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISO 2631-1 Analyzer")
        self.root.geometry("450x250")

        ttk.Button(root, text="Analyze Single CSV", command=self.analyze_file).pack(pady=10)
        ttk.Button(root, text="Analyze Folder", command=self.analyze_folder).pack(pady=10)
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, foreground="green").pack(pady=10)

    def analyze_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if filepath:
            try:
                analyzer = ISO2631Analyzer(filepath)
                res = analyzer.analyze()
                self.status.set(f"✅ Analyzed: {res.parent.name}")
            except Exception as e:
                self.status.set("❌ Error")
                messagebox.showerror("Error", str(e))

    def analyze_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            csv_files = Path(folder).rglob("*.csv")
            for file in csv_files:
                try:
                    analyzer = ISO2631Analyzer(file)
                    analyzer.analyze()
                except Exception as e:
                    print(f"⚠️ Failed on {file.name}: {e}")
            self.status.set("✅ Finished folder analysis")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()
