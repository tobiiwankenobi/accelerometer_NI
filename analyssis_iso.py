import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import firwin2, lfilter, freqz, find_peaks
import os
import glob
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import shutil
from pathlib import Path

class ISO2631Analyzer:
    def __init__(self):
        # Exact frequency weightings from ISO 2631-1 Tables 3 and 4
        self.frequency_weightings = {
            # Principal weightings (Table 3)
            'Wk': {
                'description': 'Vertical (z-axis) vibration - seat surface',
                'k_factor': 1.0,
                'weights': {
                    # Frequency (Hz): weighting factor (×1000)
                    0.1: 31.2, 0.125: 48.6, 0.16: 79.0, 0.2: 121, 0.25: 182,
                    0.315: 263, 0.4: 352, 0.5: 418, 0.63: 459, 0.8: 477,
                    1.0: 482, 1.25: 484, 1.6: 494, 2.0: 531, 2.5: 631,
                    3.15: 804, 4.0: 967, 5.0: 1039, 6.3: 1054, 8.0: 1036,
                    10.0: 988, 12.5: 902, 16.0: 768, 20.0: 636, 25.0: 513,
                    31.5: 405, 40.0: 314, 50.0: 246, 63.0: 186, 80.0: 132
                }
            },
            'Wd': {
                'description': 'Horizontal (x,y-axis) vibration - seat surface',
                'k_factor': 1.4,
                'weights': {
                    0.1: 62.4, 0.125: 97.3, 0.16: 158, 0.2: 243, 0.25: 365,
                    0.315: 530, 0.4: 713, 0.5: 853, 0.63: 944, 0.8: 992,
                    1.0: 1011, 1.25: 1008, 1.6: 968, 2.0: 890, 2.5: 776,
                    3.15: 642, 4.0: 512, 5.0: 409, 6.3: 323, 8.0: 253,
                    10.0: 212, 12.5: 161, 16.0: 125, 20.0: 100, 25.0: 80.0,
                    31.5: 63.2, 40.0: 49.4, 50.0: 38.8, 63.0: 29.5, 80.0: 21.1
                }
            },
            'Wf': {
                'description': 'Motion sickness (z-axis)',
                'k_factor': 1.0,
                'weights': {
                    0.02: 24.2, 0.025: 37.7, 0.0315: 59.7, 0.04: 97.1, 0.05: 157,
                    0.063: 267, 0.08: 461, 0.1: 695, 0.125: 895, 0.16: 1006,
                    0.2: 992, 0.25: 854, 0.315: 619, 0.4: 384, 0.5: 224,
                    0.63: 116, 0.8: 53.0, 1.0: 23.5, 1.25: 9.98, 1.6: 3.77,
                    2.0: 1.55, 2.5: 0.64, 3.15: 0.25, 4.0: 0.097
                }
            },
            # Additional weightings (Table 4)
            'Wc': {
                'description': 'Seat-back vibration',
                'k_factor': 0.8,
                'weights': {
                    0.1: 62.4, 0.125: 97.2, 0.16: 158, 0.2: 243, 0.25: 364,
                    0.315: 527, 0.4: 708, 0.5: 843, 0.63: 929, 0.8: 972,
                    1.0: 991, 1.25: 1000, 1.6: 1007, 2.0: 1012, 2.5: 1017,
                    3.15: 1022, 4.0: 1024, 5.0: 1013, 6.3: 974, 8.0: 891,
                    10.0: 776, 12.5: 647, 16.0: 512, 20.0: 409, 25.0: 325,
                    31.5: 256, 40.0: 199, 50.0: 156, 63.0: 118, 80.0: 84.4
                }
            },
            'We': {
                'description': 'Rotational vibration',
                'k_factor': 1.0,  # Note: k factors vary by axis (see standard)
                'weights': {
                    0.1: 62.5, 0.125: 97.5, 0.16: 159, 0.2: 245, 0.25: 368,
                    0.315: 536, 0.4: 723, 0.5: 862, 0.63: 939, 0.8: 941,
                    1.0: 880, 1.25: 772, 1.6: 632, 2.0: 512, 2.5: 409,
                    3.15: 323, 4.0: 253, 5.0: 202, 6.3: 160, 8.0: 125,
                    10.0: 100, 12.5: 80.1, 16.0: 62.5, 20.0: 50.0, 25.0: 39.9,
                    31.5: 31.6, 40.0: 24.7, 50.0: 19.4, 63.0: 14.8, 80.0: 10.5
                }
            },
            'Wj': {
                'description': 'Recumbent position (head)',
                'k_factor': 1.0,
                'weights': {
                    0.1: 31.0, 0.125: 48.3, 0.16: 78.5, 0.2: 120, 0.25: 181,
                    0.315: 262, 0.4: 351, 0.5: 417, 0.63: 458, 0.8: 478,
                    1.0: 484, 1.25: 485, 1.6: 483, 2.0: 482, 2.5: 489,
                    3.15: 524, 4.0: 528, 5.0: 793, 6.3: 946, 8.0: 1017,
                    10.0: 1030, 12.5: 1026, 16.0: 1018, 20.0: 1012, 25.0: 1007,
                    31.5: 1001, 40.0: 991, 50.0: 972, 63.0: 931, 80.0: 843
                }
            }
        }

    def validate_csv_structure(self, file_path):
        """Validate that the CSV has the expected structure"""
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline()
            return len(first_line.split(',')) >= 2  # At least time + one data column
        except Exception as e:
            print(f"Error validating file {file_path}: {e}")
            return False

    def design_weighting_filter(self, fs, filter_type, numtaps=501):
        """Design FIR filter based on exact ISO 2631-1 frequency weightings"""
        if filter_type not in self.frequency_weightings:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        weighting = self.frequency_weightings[filter_type]
        freqs = np.array(list(weighting['weights'].keys()))
        gains = np.array(list(weighting['weights'].values())) / 1000.0  # Convert from ×1000
    
        # Ensure frequencies are sorted in increasing order
        sort_idx = np.argsort(freqs)
        freqs = freqs[sort_idx]
        gains = gains[sort_idx]
    
        # Add 0 gain at 0 Hz and Nyquist frequency
        freqs = np.concatenate(([0], freqs, [fs/2]))
        gains = np.concatenate(([0], gains, [0]))
    
        # Verify frequencies are strictly increasing
        if not np.all(np.diff(freqs) > 0):
            raise ValueError("Filter frequencies must be strictly increasing")
    
        # Design FIR filter
        taps = firwin2(numtaps, freqs, gains, fs=fs, window='hamming')
        return taps
    
    def apply_weighting(self, data, fs, filter_type):
        """Apply frequency weighting to data"""
        taps = self.design_weighting_filter(fs, filter_type)
        # Ensure data is float64 before filtering
        data = np.asarray(data, dtype=np.float64)
        return lfilter(taps, 1.0, data)
    
    def calculate_rms(self, data):
        """Calculate RMS of the data"""
        return np.sqrt(np.mean(data ** 2))
    
    def calculate_vdv(self, data, fs):
        """Calculate Vibration Dose Value (VDV) according to ISO 2631-1"""
        return (np.sum(np.abs(data) ** 4) / fs) ** 0.25
    
    def calculate_mtv(self, data, fs, tau=1.0):
        """Calculate Maximum Transient Vibration Value (MTVV)"""
        window_size = int(tau * fs)
        squared = data ** 2
        running_rms = np.sqrt(np.convolve(squared, np.ones(window_size)/window_size, mode='valid'))
        return np.max(running_rms)
    
    def calculate_crest_factor(self, data):
        """Calculate crest factor (peak/RMS ratio)"""
        rms = self.calculate_rms(data)
        if rms == 0:
            return 0
        return np.max(np.abs(data)) / rms
    
    def plot_weighting_curves(self):
        """Plot all ISO 2631-1 frequency weighting curves"""
        plt.figure(figsize=(12, 8))
        
        # Create frequency range for plotting
        freqs = np.logspace(np.log10(0.01), np.log10(100), 500)
        
        for name, params in self.frequency_weightings.items():
            # Get exact weights from standard
            w_freqs = np.array(list(params['weights'].keys()))
            w_gains = np.array(list(params['weights'].values())) / 1000.0
            
            # Interpolate for smooth plot
            gains = np.interp(freqs, w_freqs, w_gains, left=0, right=0)
            
            plt.semilogx(freqs, 20*np.log10(gains+1e-6), 
                        label=f"{name}: {params['description']}")
        
        plt.title("ISO 2631-1 Frequency Weighting Curves")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain (dB)")
        plt.grid(True, which="both", ls="-")
        plt.legend()
        plt.xlim(0.01, 100)
        plt.ylim(-60, 10)
        plt.tight_layout()
        plt.show()
    
    def interpret_comfort(self, rms_val):
        """Interpret RMS values for comfort according to ISO 2631-1 Annex C"""
        if np.isnan(rms_val):
            return "No data", "N/A"
        elif rms_val < 0.315:
            return "Not uncomfortable", "⚪️"
        elif rms_val < 0.63:
            return "A little uncomfortable", "🟢"
        elif rms_val < 1.0:
            return "Fairly uncomfortable", "🟡"
        elif rms_val < 1.6:
            return "Uncomfortable", "🟠"
        else:
            return "Very uncomfortable", "🔴"
    
    def interpret_health(self, rms_val, duration_hours):
        """Interpret RMS values for health according to ISO 2631-1 Annex B"""
        if np.isnan(rms_val):
            return "No data", "N/A"
            
        # Convert to m/s² (assuming input is in g)
        rms_ms2 = rms_val * 9.81
        
        # Health guidance caution zone boundaries
        lower_bound = 0.5 * (duration_hours ** -0.5)  # Equation B.1
        upper_bound = 1.0 * (duration_hours ** -0.5)  # Equation B.1
        
        if rms_ms2 < lower_bound:
            return "Health effects not clearly documented", "⚪️"
        elif rms_ms2 < upper_bound:
            return "Caution - potential health risks", "🟡"
        else:
            return "Health risks are likely", "🔴"
    
    def find_dominant_frequencies(self, data, fs, n_peaks=3, min_height=0.1):
        """Find dominant frequencies in the signal"""
        n = len(data)
        if n == 0:
            return np.array([]), np.array([])
            
        # Compute FFT
        fft_vals = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(n, d=1/fs)
        
        # Find peaks
        peaks, properties = find_peaks(fft_vals, height=min_height*np.max(fft_vals))
        peak_freqs = freqs[peaks]
        peak_mags = fft_vals[peaks]
        
        # Sort by magnitude and return top n peaks
        if len(peaks) > 0:
            idx = np.argsort(peak_mags)[::-1][:n_peaks]
            return peak_freqs[idx], peak_mags[idx]
        return np.array([]), np.array([])

    def analyze_file(self, file_path, duration_hours=1.0):
        """Analyze a measurement file according to ISO 2631-1"""
        print(f"\n📂 Analyzing file: {file_path}")

        # Initialize empty return values in case of early return
        empty_df = pd.DataFrame(columns=['Channel', 'Filter', 'RMS (g)', 'Weighted RMS (g)', 
                                    'VDV (g·s^1.75)', 'MTVV (g)', 'Crest Factor', 
                                    'Dominant Freq (Hz)', 'Comfort', 'Health'])
        
        if not self.validate_csv_structure(file_path):
            print("❌ Invalid CSV structure")
            return empty_df, None, None, None

        try:
            # Read CSV while skipping comment lines
            try:
                df = pd.read_csv(file_path, encoding='utf-8', engine='python', comment='#')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp1250', engine='python', comment='#')

            # Data validation
            if len(df.columns) < 2:
                print("❌ Need at least 2 columns (time + data)")
                return empty_df, None, None, None

            # Convert to numeric and clean
            df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)

            if len(df) < 100:
                print("❌ Insufficient data points (<100)")
                return empty_df, None, None, None

            time = df.iloc[:, 0].to_numpy()
            data = df.iloc[:, 1:].to_numpy()

            # Check time intervals
            dt = np.diff(time)
            if len(dt) == 0:
                print("❌ No time intervals calculated")
                return empty_df, None, None, None

            # Handle negative time intervals
            if np.any(dt < 0):
                print("⚠️ Correcting negative time intervals")
                time = np.abs(time)
                dt = np.diff(time)

            mean_dt = np.mean(dt)
            fs = 1 / mean_dt
            
            # Check for unrealistic sampling frequency
            if fs > 10000:
                print("⚠️ Checking for time unit error (ms vs s)")
                if np.mean(time) > 1e6:
                    print("⚠️ Converting ms to s")
                    time = time / 1000
                    dt = np.diff(time)
                    mean_dt = np.mean(dt)
                    fs = 1 / mean_dt

            # Trim time data for filtered output
            time_trimmed = time[len(time)//2:]
            time_trimmed = time_trimmed[:len(time_trimmed)//2*2]

            # Assume channels are in x, y, z order
            channel_names = ['x', 'y', 'z'][:data.shape[1]]
            filter_types = ['Wd', 'Wd', 'Wk'][:data.shape[1]]

            results = []
            weighted_signals = []
            
            # Create figure for plots
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(len(channel_names), 2, width_ratios=[2, 1])

            for i, (channel_data, channel_name, filter_type) in enumerate(zip(data.T, channel_names, filter_types)):
                try:
                    # Apply frequency weighting
                    weighted_data = self.apply_weighting(channel_data, fs, filter_type)
                    weighted_data = weighted_data[len(weighted_data)//2:]
                    
                    # Align lengths
                    min_len = min(len(weighted_data), len(time_trimmed))
                    weighted_data = weighted_data[:min_len]
                    channel_time = time_trimmed[:min_len]

                    # Calculate metrics
                    rms = self.calculate_rms(weighted_data)
                    vdv = self.calculate_vdv(weighted_data, fs)
                    mtv = self.calculate_mtv(weighted_data, fs)
                    crest = self.calculate_crest_factor(weighted_data)
                    freqs, mags = self.find_dominant_frequencies(weighted_data, fs)
                    
                    # Apply k-factor
                    k_factor = self.frequency_weightings[filter_type]['k_factor']
                    weighted_rms = rms * k_factor
                    
                    # Store results
                    results.append({
                        'Channel': channel_name,
                        'Filter': filter_type,
                        'RMS (g)': rms,
                        'Weighted RMS (g)': weighted_rms,
                        'VDV (g·s^1.75)': vdv,
                        'MTVV (g)': mtv,
                        'Crest Factor': crest,
                        'Dominant Freq (Hz)': ', '.join(f"{f:.2f}" for f in freqs) if len(freqs) > 0 else 'None',
                        'Comfort': self.interpret_comfort(weighted_rms)[0],
                        'Health': self.interpret_health(weighted_rms, duration_hours)[0]
                    })
                    
                    weighted_signals.append(weighted_data)
                    
                    # Time domain plot
                    ax_time = fig.add_subplot(gs[i, 0])
                    ax_time.plot(channel_time, weighted_data, label=f'{channel_name}-axis ({filter_type})')
                    ax_time.set_title(f"{channel_name}-axis vibration ({filter_type} weighted)")
                    ax_time.set_xlabel("Time (s)")
                    ax_time.set_ylabel("Acceleration (g)")
                    ax_time.legend()
                    ax_time.grid(True)
                    
                    # Frequency domain plot
                    ax_freq = fig.add_subplot(gs[i, 1])
                    n = len(weighted_data)
                    fft_vals = np.abs(np.fft.rfft(weighted_data))
                    freqs_fft = np.fft.rfftfreq(n, d=1/fs)
                    
                    ax_freq.semilogy(freqs_fft, fft_vals + 1e-8)
                    ax_freq.set_title(f"{channel_name}-axis spectrum")
                    ax_freq.set_xlabel("Frequency (Hz)")
                    ax_freq.set_ylabel("Amplitude")
                    ax_freq.grid(True, which="both", ls="-")
                    ax_freq.set_xlim(0, 100)
                    
                    # Mark dominant frequencies
                    for freq, mag in zip(freqs, mags):
                        ax_freq.axvline(x=freq, color='r', linestyle='--', alpha=0.5)
                        ax_freq.text(freq, mag, f"{freq:.1f} Hz", 
                                ha='center', va='bottom', color='r')
                    
                except Exception as e:
                    print(f"❌ Error processing channel {channel_name}: {e}")
                    results.append({
                        'Channel': channel_name,
                        'Filter': filter_type,
                        'Error': str(e)
                    })
                    weighted_signals.append(None)

            plt.tight_layout()
            
            # Calculate vector sum if we have valid signals
            av_signal = None
            if all(s is not None for s in weighted_signals):
                weighted_array = np.array(weighted_signals)
                av = np.sqrt(np.sum(weighted_array ** 2, axis=0))
                av_rms = self.calculate_rms(av)
                av_vdv = self.calculate_vdv(av, fs)
                av_mtv = self.calculate_mtv(av, fs)
                av_freqs, av_mags = self.find_dominant_frequencies(av, fs)
                
                results.append({
                    'Channel': 'Vector Sum',
                    'Filter': 'N/A',
                    'RMS (g)': av_rms,
                    'Weighted RMS (g)': av_rms,
                    'VDV (g·s^1.75)': av_vdv,
                    'MTVV (g)': av_mtv,
                    'Crest Factor': self.calculate_crest_factor(av),
                    'Dominant Freq (Hz)': ', '.join(f"{f:.2f}" for f in av_freqs) if len(av_freqs) > 0 else 'None',
                    'Comfort': self.interpret_comfort(av_rms)[0],
                    'Health': 'N/A (use individual axes for health)'
                })
                av_signal = av
            else:
                print("⚠️ Could not calculate vector sum due to missing channels")

            results_df = pd.DataFrame(results)
            return results_df, av_signal, fig, time_trimmed

        except Exception as e:
            print(f"❌ Error analyzing file: {e}")
            return empty_df, None, None, None


class VibrationAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISO 2631-1 Vibration Analysis")
        self.root.geometry("1000x800")
        
        self.analyzer = ISO2631Analyzer()
        self.current_folder = None
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Folder selection section
        self.folder_frame = ttk.LabelFrame(self.main_frame, text="Select Measurement Folder", padding="10")
        self.folder_frame.pack(fill=tk.X, pady=5)
        
        self.folder_path = tk.StringVar()
        self.folder_entry = ttk.Entry(self.folder_frame, textvariable=self.folder_path, width=70)
        self.folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.browse_button = ttk.Button(self.folder_frame, text="Browse", command=self.browse_folder)
        self.browse_button.pack(side=tk.RIGHT)
        
        # Seat selection
        self.seat_frame = ttk.LabelFrame(self.main_frame, text="Select Seat Position", padding="10")
        self.seat_frame.pack(fill=tk.X, pady=5)
        
        self.seat_var = tk.StringVar()
        self.seat_combobox = ttk.Combobox(self.seat_frame, textvariable=self.seat_var, state="readonly")
        self.seat_combobox.pack(fill=tk.X)
        
        # File list with scrollbar
        self.file_frame = ttk.LabelFrame(self.main_frame, text="Measurement Files", padding="10")
        self.file_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.file_listbox = tk.Listbox(self.file_frame, selectmode=tk.SINGLE)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.scrollbar = ttk.Scrollbar(self.file_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=self.scrollbar.set)
        
        # Analysis options
        self.options_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options", padding="10")
        self.options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.options_frame, text="Duration (hours):").grid(row=0, column=0, sticky=tk.W)
        self.duration_entry = ttk.Entry(self.options_frame, width=10)
        self.duration_entry.insert(0, "1.0")
        self.duration_entry.grid(row=0, column=1, sticky=tk.W)
        
        # Buttons frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        self.analyze_button = ttk.Button(self.button_frame, text="Analyze Selected File", 
                                       command=self.analyze_selected_file)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        self.analyze_all_button = ttk.Button(self.button_frame, text="Analyze All Files in Seat", 
                                           command=self.analyze_all_files)
        self.analyze_all_button.pack(side=tk.LEFT, padx=5)
        
        self.plot_weightings_button = ttk.Button(self.button_frame, text="Show Weighting Curves", 
                                               command=self.analyzer.plot_weighting_curves)
        self.plot_weightings_button.pack(side=tk.LEFT, padx=5)
        
        self.quit_button = ttk.Button(self.button_frame, text="Quit", command=root.quit)
        self.quit_button.pack(side=tk.RIGHT, padx=5)
        
        # Results display
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Analysis Results", padding="10")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD)
        self.scrollbar_results = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.scrollbar_results.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=self.scrollbar_results.set)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize
        self.folder_path.trace_add('write', self.folder_changed)
    
    def browse_folder(self):
        """Open folder selection dialog"""
        folder_selected = filedialog.askdirectory(initialdir="csv_pomiary")
        if folder_selected:
            self.folder_path.set(folder_selected)
    
    def folder_changed(self, *args):
        """Handle folder path change"""
        folder_path = self.folder_path.get()
        if os.path.isdir(folder_path):
            self.current_folder = folder_path
            self.update_seat_list()
            self.update_file_list()
    
    def update_seat_list(self):
        """Update list of available seat positions"""
        if not self.current_folder:
            return
            
        seat_folders = []
        for item in os.listdir(self.current_folder):
            if os.path.isdir(os.path.join(self.current_folder, item)) and item.startswith("seat"):
                seat_folders.append(item)
        
        self.seat_combobox['values'] = sorted(seat_folders)
        if seat_folders:
            self.seat_var.set(seat_folders[0])
    
    def update_file_list(self):
        """Update list of measurement files for selected seat"""
        self.file_listbox.delete(0, tk.END)
        
        if not self.current_folder or not self.seat_var.get():
            return
            
        seat_folder = os.path.join(self.current_folder, self.seat_var.get())
        if not os.path.isdir(seat_folder):
            return
            
        measurement_files = []
        for root, dirs, files in os.walk(seat_folder):
            for file in files:
                if file.endswith(".csv") and file.startswith("measurement_"):
                    measurement_files.append(os.path.join(root, file))
        
        # Sort files by measurement number
        measurement_files.sort(key=lambda x: int(x.split('_')[-2]))
        
        for file_path in measurement_files:
            self.file_listbox.insert(tk.END, os.path.relpath(file_path, self.current_folder))
    
    def analyze_selected_file(self):
        """Analyze the currently selected file"""
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a file to analyze")
            return
            
        selected_file = self.file_listbox.get(selection[0])
        file_path = os.path.join(self.current_folder, selected_file)
        
        try:
            duration_hours = float(self.duration_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for duration")
            return
            
        self.analyze_file(file_path, duration_hours)
    
    def analyze_all_files(self):
        """Analyze all files in the selected seat folder"""
        if not self.current_folder or not self.seat_var.get():
            messagebox.showerror("Error", "Please select a folder and seat position")
            return
            
        seat_folder = os.path.join(self.current_folder, self.seat_var.get())
        if not os.path.isdir(seat_folder):
            messagebox.showerror("Error", "Selected seat folder does not exist")
            return
            
        try:
            duration_hours = float(self.duration_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for duration")
            return
            
        # Create output folder structure
        analysis_root = os.path.join(os.path.dirname(self.current_folder), 
                                    os.path.basename(self.current_folder) + "_analiza")
        seat_analysis_folder = os.path.join(analysis_root, self.seat_var.get())
        os.makedirs(seat_analysis_folder, exist_ok=True)
        
        # Find all measurement files
        measurement_files = []
        for root, dirs, files in os.walk(seat_folder):
            for file in files:
                if file.endswith(".csv") and file.startswith("measurement_"):
                    measurement_files.append(os.path.join(root, file))
        
        if not measurement_files:
            messagebox.showinfo("Info", "No measurement files found in selected seat folder")
            return
            
        # Sort files by measurement number
        measurement_files.sort(key=lambda x: int(x.split('_')[-2]))
        
        # Analyze each file
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Analyzing all files in: {self.seat_var.get()}\n\n")
        
        for file_path in measurement_files:
            try:
                # Get relative path for output folder structure
                rel_path = os.path.relpath(os.path.dirname(file_path), self.current_folder)
                measurement_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Create corresponding output folder
                output_folder = os.path.join(analysis_root, rel_path, measurement_name + "_analysis")
                os.makedirs(output_folder, exist_ok=True)
                
                self.results_text.insert(tk.END, f"Analyzing: {os.path.basename(file_path)}\n")
                self.results_text.update()
                
                # Perform analysis
                results_df, av_signal, fig, time_data = self.analyzer.analyze_file(file_path, duration_hours)
                
                # Save results
                output_csv = os.path.join(output_folder, f"{measurement_name}_results.csv")
                results_df.to_csv(output_csv, index=False)
                
                # Save plots
                output_pdf = os.path.join(output_folder, f"{measurement_name}_plots.pdf")
                fig.savefig(output_pdf, format='pdf', bbox_inches='tight')
                
                output_svg = os.path.join(output_folder, f"{measurement_name}_plots.svg")
                fig.savefig(output_svg, format='svg', bbox_inches='tight')
                
                # Generate comprehensive report
                output_report = os.path.join(output_folder, f"{measurement_name}_report.pdf")
                with PdfPages(output_report) as pdf:
                    # Title page
                    plt.figure(figsize=(8.27, 11.69))
                    plt.text(0.5, 0.9, "ISO 2631-1 Vibration Analysis Report", 
                            ha='center', va='center', fontsize=16)
                    plt.text(0.5, 0.85, f"File: {measurement_name}", 
                            ha='center', va='center', fontsize=12)
                    plt.text(0.5, 0.8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                            ha='center', va='center', fontsize=10)
                    plt.axis('off')
                    pdf.savefig(bbox_inches='tight')
                    plt.close()
                    
                    # Results table
                    fig_table, ax = plt.subplots(figsize=(8.27, 11.69))
                    ax.axis('off')
                    table_data = results_df.fillna("").values.tolist()
                    col_labels = results_df.columns.tolist()
                    table = ax.table(cellText=table_data, colLabels=col_labels, 
                                   loc='center', cellLoc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1, 1.5)
                    pdf.savefig(fig_table, bbox_inches='tight')
                    plt.close(fig_table)
                    
                    # Analysis plots
                    pdf.savefig(fig, bbox_inches='tight')
                    
                    # Additional spectral analysis
                    if av_signal is not None:
                        try:
                            n = len(av_signal)
                            fs = 1 / (time_data[1] - time_data[0])
                            fft_vals = np.abs(np.fft.rfft(av_signal))
                            freqs = np.fft.rfftfreq(n, d=1/fs)
                            
                            fig_spectrum = plt.figure(figsize=(10, 5))
                            plt.semilogy(freqs, fft_vals + 1e-8)
                            plt.title("FFT of Weighted Vector Sum (av)")
                            plt.xlabel("Frequency (Hz)")
                            plt.ylabel("Amplitude")
                            plt.grid(True, which="both")
                            plt.xlim(0, 100)
                            pdf.savefig(fig_spectrum, bbox_inches='tight')
                            plt.close(fig_spectrum)
                        except Exception as e:
                            self.results_text.insert(tk.END, f"⚠️ Could not generate spectrum plot: {e}\n")
                
                self.results_text.insert(tk.END, f"  Results saved to: {output_folder}\n")
                plt.close('all')
                
            except Exception as e:
                self.results_text.insert(tk.END, f"❌ Error analyzing {os.path.basename(file_path)}: {str(e)}\n")
                continue
        
        self.results_text.insert(tk.END, "\nAnalysis complete!\n")
        messagebox.showinfo("Analysis Complete", f"All files analyzed. Results saved in:\n{analysis_root}")
    
    def analyze_file(self, file_path, duration_hours):
        """Analyze a single file and display results"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Analyzing: {os.path.basename(file_path)}\n")
        
        if not self.analyzer.validate_csv_structure(file_path):
            messagebox.showerror("Error", "Invalid CSV structure - need time column and at least one data column")
            return
            
        try:
            duration_hours = float(self.duration_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for duration")
            return
            
        try:
            # Perform analysis
            results_df, av_signal, fig, time_data = self.analyzer.analyze_file(file_path, duration_hours)
            
            # Create output directory structure
            rel_path = os.path.relpath(os.path.dirname(file_path), self.current_folder)
            measurement_name = os.path.splitext(os.path.basename(file_path))[0]
            
            analysis_root = os.path.join(os.path.dirname(self.current_folder), 
                                        os.path.basename(self.current_folder) + "_analiza")
            output_folder = os.path.join(analysis_root, rel_path, measurement_name + "_analysis")
            os.makedirs(output_folder, exist_ok=True)
            
            # Save results as CSV
            output_csv = os.path.join(output_folder, f"{measurement_name}_results.csv")
            results_df.to_csv(output_csv, index=False)
            self.results_text.insert(tk.END, f"\nResults saved to: {output_csv}\n")
            
            # Save plots in vector formats
            output_pdf = os.path.join(output_folder, f"{measurement_name}_plots.pdf")
            fig.savefig(output_pdf, format='pdf', bbox_inches='tight')
            self.results_text.insert(tk.END, f"Plots saved to PDF: {output_pdf}\n")
            
            output_svg = os.path.join(output_folder, f"{measurement_name}_plots.svg")
            fig.savefig(output_svg, format='svg', bbox_inches='tight')
            self.results_text.insert(tk.END, f"Plots saved to SVG: {output_svg}\n")
            
            # Generate comprehensive PDF report
            output_report = os.path.join(output_folder, f"{measurement_name}_report.pdf")
            with PdfPages(output_report) as pdf:
                # Title page
                plt.figure(figsize=(8.27, 11.69))
                plt.text(0.5, 0.9, "ISO 2631-1 Vibration Analysis Report", 
                        ha='center', va='center', fontsize=16)
                plt.text(0.5, 0.85, f"File: {measurement_name}", 
                        ha='center', va='center', fontsize=12)
                plt.text(0.5, 0.8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                        ha='center', va='center', fontsize=10)
                plt.axis('off')
                pdf.savefig(bbox_inches='tight')
                plt.close()
                
                # Results table
                fig_table, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis('off')
                table_data = results_df.fillna("").values.tolist()
                col_labels = results_df.columns.tolist()
                table = ax.table(cellText=table_data, colLabels=col_labels, 
                               loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.5)
                pdf.savefig(fig_table, bbox_inches='tight')
                plt.close(fig_table)
                
                # Analysis plots
                pdf.savefig(fig, bbox_inches='tight')
                
                # Additional spectral analysis
                if av_signal is not None:
                    try:
                        n = len(av_signal)
                        fs = 1 / (time_data[1] - time_data[0])
                        fft_vals = np.abs(np.fft.rfft(av_signal))
                        freqs = np.fft.rfftfreq(n, d=1/fs)
                        
                        fig_spectrum = plt.figure(figsize=(10, 5))
                        plt.semilogy(freqs, fft_vals + 1e-8)
                        plt.title("FFT of Weighted Vector Sum (av)")
                        plt.xlabel("Frequency (Hz)")
                        plt.ylabel("Amplitude")
                        plt.grid(True, which="both")
                        plt.xlim(0, 100)
                        pdf.savefig(fig_spectrum, bbox_inches='tight')
                        plt.close(fig_spectrum)
                    except Exception as e:
                        self.results_text.insert(tk.END, f"⚠️ Could not generate spectrum plot: {e}\n")
            
            self.results_text.insert(tk.END, f"Full report saved to: {output_report}\n")
            
            # Show summary in results text
            self.results_text.insert(tk.END, "\nAnalysis Summary:\n")
            self.results_text.insert(tk.END, results_df.to_string())
            
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during analysis:\n{str(e)}")
            self.results_text.insert(tk.END, f"\n❌ Error: {str(e)}\n")
        finally:
            plt.close('all')

def main():
    root = tk.Tk()
    app = VibrationAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()