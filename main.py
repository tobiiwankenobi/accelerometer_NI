import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
import time
from nidaqmx.constants import TerminalConfiguration
import threading
import queue
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

class AccelerometerApp:
    import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
import time
from nidaqmx.constants import TerminalConfiguration
import threading
import queue
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

class AccelerometerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Accelerometer Measurement System")
        self.root.geometry("600x500")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.plot_queue = queue.Queue()
        self.status_queue = queue.Queue()

        self.device = 'cDAQ1Mod1'
        self.num_channels = 4
        self.sampling_rate = 10000
        self.available_durations = [round(m, 1) for m in np.arange(1, 10.5, 1)]  # # 0.5–10 min co 0.5
        self.duration = 60
        self.sensitivities = [0.01051, 0.01076, 0.01055, 0.01000]

        self.car_name = ""
        self.seat_position = 0
        self.measurement_numbers = {1: 0, 2: 0, 3: 0}
        self.measurement_active = False
        self.daq_initialized = False

        self.create_widgets()
        self.root.after(100, self.process_queues)
        threading.Thread(target=self.initialize_daq, daemon=True).start()
    
    def create_widgets(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing application...")
        ttk.Label(
            self.main_frame, 
            textvariable=self.status_var, 
            wraplength=550,
            font=('Arial', 10)
        ).grid(row=0, column=0, columnspan=4, pady=5, sticky=tk.W)
        
        # Car name entry
        ttk.Label(self.main_frame, text="Car Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.car_entry = ttk.Entry(self.main_frame, width=25)
        self.car_entry.grid(row=1, column=1, columnspan=3, padx=5, sticky=tk.W)
        
        # Seat position selection
        ttk.Label(self.main_frame, text="Seat Position:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.seat_var = tk.IntVar(value=1)
        for i in range(3):
            ttk.Radiobutton(
                self.main_frame, 
                text=f"Seat {i+1}", 
                variable=self.seat_var, 
                value=i+1
            ).grid(row=2, column=i+1, sticky=tk.W, padx=5)
        
        # Measurement duration selection
        ttk.Label(self.main_frame, text="Duration (min):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.duration_var = tk.IntVar(value=self.duration)
        self.duration_menu = ttk.Combobox(
            self.main_frame,
            textvariable=self.duration_var,
            values=self.available_durations,
            width=5,
            state="readonly"
        )
        self.duration_menu.grid(row=3, column=1, sticky=tk.W)
        self.duration_menu.bind("<<ComboboxSelected>>", self.update_duration)
        
        # Measurement button
        self.measure_btn = ttk.Button(
            self.main_frame, 
            text="Start Measurement", 
            command=self.start_measurement,
            state=tk.DISABLED
        )
        self.measure_btn.grid(row=4, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        # Quit button
        self.quit_btn = ttk.Button(
            self.main_frame,
            text="Quit",
            command=self.on_closing
        )
        self.quit_btn.grid(row=4, column=2, columnspan=2, pady=10, sticky=tk.E)
        
        # Measurement info display
        self.info_frame = ttk.LabelFrame(self.main_frame, text="Measurement Info", padding=10)
        self.info_frame.grid(row=5, column=0, columnspan=4, pady=10, sticky=tk.NSEW)
        
        self.info_text = tk.Text(
            self.info_frame, 
            height=6, 
            width=60, 
            state=tk.DISABLED,
            font=('Arial', 9)
        )
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.main_frame.rowconfigure(5, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        
        self.update_info()
    
    def process_queues(self):
        """Handle messages from other threads"""
        try:
            # Process status updates
            while not self.status_queue.empty():
                message = self.status_queue.get_nowait()
                self.status_var.set(message)
                self.root.update()
            
            # Process plot requests
            while not self.plot_queue.empty():
                data = self.plot_queue.get_nowait()
                self.create_plot(*data)
        finally:
            self.root.after(100, self.process_queues)
    
    def update_duration(self, event=None):
        """Update measurement duration from combobox selection"""
        self.duration = int(float(self.duration_var.get()) * 60)
        self.status_var.set(f"Measurement duration set to {self.duration} seconds")
    
    def initialize_daq(self):
        """Initialize DAQ and discard initial readings"""
        try:
            self.status_queue.put("Initializing DAQ...")
            
            with nidaqmx.Task() as init_task:
                # Configure all channels
                for i in range(self.num_channels):
                    init_task.ai_channels.add_ai_voltage_chan(
                        f"{self.device}/ai{i}",
                        terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                        min_val=-5.0,
                        max_val=5.0
                    )
                
                # Discard initial readings (2 seconds)
                self.status_queue.put("Discarding initial readings (2 seconds)...")
                
                discard_samples = int(self.sampling_rate * 2)
                init_task.timing.cfg_samp_clk_timing(
                    rate=self.sampling_rate, 
                    samps_per_chan=discard_samples
                )
                init_task.start()
                time.sleep(0.1)
                # Read and discard initial samples
                init_task.read(number_of_samples_per_channel=discard_samples)
                init_task.stop()
            
            self.daq_initialized = True
            self.status_queue.put("DAQ initialized and ready")
            self.root.after(0, lambda: self.measure_btn.config(state=tk.NORMAL))
        except Exception as e:
            self.status_queue.put(f"DAQ initialization failed: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror(
                "Error", 
                f"Failed to initialize DAQ: {str(e)}\n\nPlease check your NI-DAQmx installation and device connections.")
            )
    
    def update_info(self):
        """Update measurement information display"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        if self.car_name:
            info = f"Current Car: {self.car_name}\n"
            info += f"Next Measurement Duration: {self.duration // 60} min\n\n"
            
            for seat in range(1, 4):
                count = self.measurement_numbers.get(seat, 0)
                remaining = max(0, 10 - count)
                info += f"Seat {seat}: {count} measurements taken ({remaining} remaining)\n"
            
            self.info_text.insert(tk.END, info)
        else:
            self.info_text.insert(tk.END, "No car selected\n\nPlease enter car name and seat position")
        
        self.info_text.config(state=tk.DISABLED)
    
    def countdown_timer(self, seconds):
        """Countdown timer that updates the GUI"""
        for i in range(seconds, 0, -1):
            if not self.measurement_active:
                break
            self.status_queue.put(f"⏳ Measurement in progress... {i} seconds remaining")
            time.sleep(1)
        
        if self.measurement_active:
            self.status_queue.put("⏰ Measurement complete!")
    
    def get_next_measurement_number(self, seat_folder):
        """Get next available measurement number for seat position"""
        try:
            existing = [f.name for f in seat_folder.glob('measurement_*.csv')]
            if not existing:
                return 1
            
            numbers = []
            for f in existing:
                try:
                    num = int(f.split('_')[1].split('.')[0])
                    numbers.append(num)
                except (IndexError, ValueError):
                    continue
            
            if not numbers:
                return 1
            
            max_num = max(numbers)
            return max_num + 1 if max_num < 10 else None
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error", 
                f"Error checking existing measurements: {str(e)}")
            )
            return None
    
    def run_measurement(self):
        """Perform the actual measurement"""
        try:
            self.measurement_active = True
            self.samples = int(self.sampling_rate * self.duration)
            
            # Validate inputs
            if not self.car_name or not self.seat_position:
                raise ValueError("Car name and seat position must be specified")
            
            # Create folder structure
            base_dir = Path("csv_pomiary")
            car_dir = base_dir / self.car_name
            seat_dir = car_dir / f"seat {self.seat_position}"
            seat_dir.mkdir(parents=True, exist_ok=True)
            
            # Check for available measurement slot
            measurement_num = self.get_next_measurement_number(seat_dir)
            if measurement_num is None:
                self.status_queue.put("Maximum 10 measurements reached for this seat position")
                return
            
            # Create DAQ task
            with nidaqmx.Task() as task:
                # Configure all channels
                for i in range(self.num_channels):
                    task.ai_channels.add_ai_voltage_chan(
                        f"{self.device}/ai{i}",
                        terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                        min_val=-5.0,
                        max_val=5.0
                    )
                
                # Configure timing
                task.timing.cfg_samp_clk_timing(
                    rate=self.sampling_rate,
                    samps_per_chan=self.samples
                )
                
                # Start countdown in a thread
                timer_thread = threading.Thread(target=self.countdown_timer, args=(self.duration,))
                timer_thread.start()
                
                # Measure time and start acquisition
                start_time = time.time()
                task.start()
                task.wait_until_done(timeout=self.duration + 5)
                
                # Read all data
                data = task.read(number_of_samples_per_channel=self.samples)
                data = np.array(data)
                end_time = time.time()
                
                # Verify data shape
                if data.shape[0] != self.num_channels:
                    raise ValueError(f"Expected {self.num_channels} channels, got {data.shape[0]}")
                
                # Wait for countdown to finish
                timer_thread.join()
                
                elapsed_time = end_time - start_time
                self.status_queue.put(f"✅ Measurement complete! Time: {elapsed_time:.2f} s")
            
            # Convert voltage to acceleration (g)
            acc_data = np.zeros_like(data)
            for i in range(self.num_channels):
                acc_data[i] = data[i] / self.sensitivities[i]
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"measurement_{measurement_num:02d}_{timestamp}.csv"
            file_path = seat_dir / filename
            
            # Prepare data for saving
            time_vector = np.arange(self.samples) / self.sampling_rate
            output_data = np.vstack((time_vector, acc_data)).T
            
            # Column headers
            header = (f"Car: {self.car_name}, Seat: {self.seat_position}, Duration: {self.duration}s\n"
                     f"Time (s)," + ",".join([f"Channel {i+1} (g)" for i in range(self.num_channels)]))
            
            # Save data to CSV
            np.savetxt(file_path, output_data, delimiter=',', header=header, comments='')
            
            # Update measurement count
            self.measurement_numbers[self.seat_position] = measurement_num
            self.root.after(0, self.update_info)
            
            # Show completion message
            self.root.after(0, lambda: messagebox.showinfo(
                "Measurement Complete", 
                f"Successfully saved measurement {measurement_num}:\n"
                f"Car: {self.car_name}\n"
                f"Seat: {self.seat_position}\n"
                f"Duration: {self.duration} seconds\n"
                f"File: {file_path}"
            ))
            
            # Add plot to queue
            #self.plot_queue.put((time_vector, acc_data, measurement_num))
            
        except Exception as e:
            error_message = str(e)
            self.root.after(0, lambda: messagebox.showerror(
                "Measurement Error", 
                 f"Error during measurement:\n{error_message}")
        )

        finally:
            self.measurement_active = False
            self.root.after(0, self.enable_ui)
    
    def create_plot(self, time_vector, acc_data, measurement_num):
        """Create plot in main thread"""
        plt.figure(figsize=(12, 8))
        
        # Plot each channel
        for i in range(self.num_channels):
            plt.subplot(self.num_channels, 1, i+1)
            plt.plot(time_vector, acc_data[i])
            plt.title(f'Channel {i+1} - Acceleration (g) {"(Noise Monitoring)" if i == 3 else ""}')
            plt.xlabel('Time (s)')
            plt.ylabel('g')
            plt.grid(True)
        
        # Main title
        plt.suptitle(
            f"{self.car_name} - Seat {self.seat_position}\n"
            f"Measurement {measurement_num} ({self.duration} seconds)",
            y=1.02
        )
        
        plt.tight_layout()
        plt.show(block=False)  # Non-blocking show
    
    def enable_ui(self):
        """Re-enable UI after measurement"""
        self.measure_btn.config(state=tk.NORMAL)
        self.car_entry.config(state=tk.NORMAL)
        self.duration_menu.config(state="readonly")
        for child in self.main_frame.winfo_children():
            if isinstance(child, ttk.Radiobutton):
                child.config(state=tk.NORMAL)
    
    def start_measurement(self):
        """Start a new measurement with current settings"""
        if not self.daq_initialized:
            messagebox.showerror("Error", "DAQ not initialized. Please wait or restart the application.")
            return
        
        self.car_name = self.car_entry.get().strip()
        self.seat_position = self.seat_var.get()
        
        if not self.car_name:
            messagebox.showerror("Error", "Please enter a car name")
            return
        
        # Disable UI during measurement
        self.measure_btn.config(state=tk.DISABLED)
        self.car_entry.config(state=tk.DISABLED)
        self.duration_menu.config(state=tk.DISABLED)
        for child in self.main_frame.winfo_children():
            if isinstance(child, ttk.Radiobutton):
                child.config(state=tk.DISABLED)
        
        # Run measurement in a separate thread
        threading.Thread(target=self.run_measurement, daemon=True).start()
    
    def on_closing(self):
        """Handle window closing"""
        if self.measurement_active:
            if messagebox.askokcancel(
                "Quit", 
                "Measurement in progress!\n"
                "Are you sure you want to quit?\n"
                "Current measurement will be incomplete."
            ):
                self.measurement_active = False
                time.sleep(0.5)  # Give thread a moment to stop
                self.root.destroy()
        else:
            if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
                self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AccelerometerApp(root)
    root.mainloop()