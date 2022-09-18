# ************************************************************************************* #
#                                                                                       #
# AudioPlot 2022                                                                        #
#                                                                                       #
# Author: Marcin Lesniak                                                                #
#                                                                                       #
# Install dependencies with:                                                            #
# pip install --upgrade numpy matplotlib soundfile sounddevice                          #
#                                                                                       #
# ************************************************************************************* #

import os
import soundfile as sf
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, Menu
from tkinter.messagebox import showinfo
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import sounddevice as sd
import time
from threading import Thread, Event
from queue import LifoQueue
from typing import Union, Any
from dataclasses import dataclass


#================= Classes ===================

class Constants:
    CURR_DIR = os.getcwd()
    BG_COLOR = "#e6ddee"
    PLOT_COLOR = "#002699"
    

class AsyncPlay(Thread):
    '''
    Thread managing sound replay
    The use of sounddevice is described here: 
    https://python-sounddevice.readthedocs.io/en/0.4.5/examples.html#play-a-sound-file
    '''

    def __init__(self, data: np.ndarray, fs: int, queue: LifoQueue, stop_event: Event, pos: int = 0):
        super().__init__()
        self.current_frame = int(pos * fs)
        self.data = data
        self.fs = fs
        self.queue = queue
        self.stop_event = stop_event
        self.curr_time = 0
    
    def run(self):
        try:
            def callback(outdata, frames, time, status):
                if status:
                    print(status)
                chunksize = min(len(self.data) - self.current_frame, frames)
                outdata[:chunksize] = self.data[self.current_frame:self.current_frame + chunksize]
                if chunksize < frames:
                    outdata[chunksize:] = 0
                    raise sd.CallbackStop()
                if self.queue.full():
                    self.queue.queue.clear()
                
                # put current time in queue so that plot can be updated
                self.curr_time = self.current_frame/self.fs
                self.queue.put(self.curr_time)
                self.current_frame += chunksize
            
            stream = sd.OutputStream(samplerate=self.fs, channels=self.data.shape[1], callback=callback, finished_callback=self.stop_event.set)
            with stream:
                self.stop_event.wait()  # Wait until playback is finished
        except Exception as e:
            print(f"An exception occured: {e}")


@dataclass
class SoundPosition:
    '''Stores info about sound file position'''

    pos: float # current position
    last_pos: float # last position
    marker_pos: Union[float,None] # marker position


@dataclass
class DirData:
    '''Stores info about files and directories'''

    directory: str
    file_name : str = ""


class App(tk.Tk):
    '''Main window of tkinter app'''

    def __init__(self):
        super().__init__()
        self.protocol('WM_DELETE_WINDOW', self.on_close)
        
        # set up events to exhange info between threads
        self.close_event = Event()
        self.pause_event = Event()
        self.stop_event = Event()

        # set up queue to exchage data between threads
        self.queue = LifoQueue(maxsize = 1)

        # objects to store data
        self.sound = SoundPosition(0.0, 0.0, None)
        self.dir = DirData(os.getcwd())

        # configure the root window
        self.title('AudioPlot')
        self.geometry('1200x600')
        self.configure(bg=Constants.BG_COLOR)
        self.resizable(0, 0) # resizing may cause problems with plot canvas blitting, so it's blocked

        # menu
        self.menubar = Menu(self)
        self.config(menu=self.menubar)
        self.file_menu = Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label='Open file...', command=self.open_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label='Exit', command=self.on_close)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        # configure grid
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.columnconfigure(4, weight=3)
        self.rowconfigure(2, minsize=400)

        # set style for all ttk widgets
        style = ttk.Style()
        style.configure('.', font=('Consolas', 11))
        
        # labels
        self.label_file = ttk.Label(self, text='File', background=Constants.BG_COLOR)
        self.label_file.grid(column=0, row=0, sticky=tk.W, padx=1, pady=1, columnspan=4)

        self.label_time = ttk.Label(self, text=f"Time: {self.sound.pos:.3f} sec.", background=Constants.BG_COLOR)
        self.label_time.grid(column=4, row=0, sticky=tk.W, padx=5, pady=5)

        self.label_marker = ttk.Label(self, text=f"Marker: - sec.", foreground='green', background=Constants.BG_COLOR)
        self.label_marker.grid(column=4, row=1, sticky=tk.W, padx=5, pady=5)

        # buttons
        self.play_button = ttk.Button(self, text='> START')
        self.play_button['command'] = self.click_play
        self.play_button.grid(column=1, row=3, sticky=tk.EW, padx=2, pady=2, ipady=10)
        self.play_button['state'] = "disabled"

        self.pause_button = ttk.Button(self, text='|| PAUSE')
        self.pause_button['command'] = self.click_pause
        self.pause_button.grid(column=2, row=3, sticky=tk.EW, padx=2, pady=2, ipady=10)
        self.pause_button['state'] = "disabled"

        self.stop_button = ttk.Button(self, text='[] STOP')
        self.stop_button['command'] = self.click_stop
        self.stop_button.grid(column=3, row=3, sticky=tk.EW, padx=2, pady=2, ipady=10)
        self.stop_button['state'] = "disabled"

        # matplotlib figure
        self.figure = Figure(figsize=(12, 4), dpi=100)
        self.figure.suptitle('Audio plot', fontsize=14)
        self.figure.set_facecolor(Constants.BG_COLOR)
        plt.style.use('seaborn-darkgrid')

    
    #========= Menu and buttons commands ============

    def open_file(self) -> None:
        '''Select audio file from disk'''

        filetypes = (
            ('audio files', '*.wav'),
            ('All files', '*.*')
        )

        file_path = filedialog.askopenfilename(title='Open audio file', initialdir=self.dir.directory, filetypes=filetypes)
        if file_path:
            self.extract_audio_data(file_path)
  

    # ========== Audio control buttons commands ============

    def click_play(self) -> None:
        '''Called when Play button is clicked'''

        # clear data
        self.queue.queue.clear()
        self.stop_event.clear()
        self.pause_event.clear()

        # start play thread
        self.play_thread = AsyncPlay(self.data, self.samplerate, self.queue, self.stop_event, self.sound.last_pos)
        self.play_thread.start()

        self.play_button['state'] = "disabled"
        
        # start updating plot of audio data
        self.update_plot(self.sound.last_pos)


    def click_pause(self) -> None:
        '''Called when Pause button is clicked'''

        if self.pause_event.is_set():
            self.pause_event.clear()
            self.stop_event.clear()
            # start play thread if sound was paused and start updating plot 
            self.play_thread = AsyncPlay(self.data, self.samplerate, self.queue, self.stop_event, self.sound.last_pos)
            self.play_thread.start()
            self.update_plot(self.sound.last_pos)
        else:
            self.pause_event.set()
            self.stop_event.set()


    def click_stop(self) -> None:
        '''Called when Stop button is clicked'''

        self.stop_event.set()
        self.pause_event.clear()
        self.sound.last_pos = 0
        self.update_plot(self.sound.last_pos)


    # ============== Sound and data events ===============

    def on_mouse_click(self, event) -> None:
        '''Called when mouse button is clicked'''

        x_cursor = event.xdata
        y_cursor = event.ydata
        if (x_cursor is not None) and (y_cursor is not None):
            # left button to change do position of audio data
            if event.button is MouseButton.LEFT:
                self.sound.last_pos = x_cursor if self.time_axis[0] <= x_cursor <= self.time_axis[-1] else self.sound.last_pos
            # right button to change the position of marker
            elif event.button is MouseButton.RIGHT:
                self.sound.marker_pos = x_cursor if self.time_axis[0] <= x_cursor <= self.time_axis[-1] else self.sound.last_pos
                self.mark_line.set_alpha(1)
                if self.sound.marker_pos != None:
                    self.label_marker['text'] = f"Marker: {self.sound.marker_pos:.3f} sec."
            
            self.pause_event.set()
            self.stop_event.set()
            
            # redraw the plot to show vertical lines (current audio position and marker)
            self.redraw(self.figure, self.figure_canvas, self.bg, [self.anim_line, self.mark_line], self.ax, [self.sound.last_pos, self.sound.marker_pos])
            self.label_time['text'] = f"Time: {self.sound.last_pos:.3f} sec."


    def extract_audio_data(self, file_path) -> None:
        '''Open audio file and read sound data'''

        if file_path:
            # try to extract data from the file
            self.data, self.samplerate = sf.read(file_path, always_2d=True)
            
            # start updating window and plotting if data is OK
            if self.data.any() and self.samplerate:
                self.stop_event.clear()
                self.pause_event.clear()
                self.dir.file_name = os.path.basename(file_path)

                showinfo(
                    title='Chosen audio file',
                    message = f"{self.dir.file_name}\nTime: {len(self.data) / self.samplerate:.3f};"
                )

                self.label_file['text'] = file_path
                self.sound.marker_pos = None
                self.sound.pos = self.sound.last_pos = 0
                
                self.plot_data()
                self.set_buttons_state("normal")
                self.update_plot(self.sound.last_pos)
        
    
    # ============= Plot methods ============
    
    def plot_data(self) -> None:
        '''Create plot of audio data'''

        stereo = True if self.data.shape[1]==2 else False
        n_data = len(self.data)

        self.ch1 = np.array([self.data[i][0] for i in range(n_data)])  # extract channel 1
        if stereo:
            self.ch2 = np.array([self.data[i][1] for i in range(n_data)])  # extaract channel 2
        else:
            self.ch2 = np.array([])
        
        self.time_axis = np.linspace(0, n_data / self.samplerate, n_data, endpoint=False)

        # make sure that figure is empty
        self.figure.clear()

        # create axes and add artists which will constitute background
        self.ax = self.figure.add_subplot(1, 1, 1, label="plot")
        self.ax.plot(self.time_axis, self.ch1, color=Constants.PLOT_COLOR, alpha=0.6)
        self.ax.axhline(y=0, xmin=self.time_axis[0], xmax=self.time_axis[-1], color='k', linewidth=0.5, alpha=0.3)
        
        # create artists which will be added later and animated
        self.anim_line = self.ax.axvline(x=0, ymin=0, ymax=1, color='r', linewidth=1.0, animated=True)
        self.mark_line = self.ax.axvline(x=0, ymin=0, ymax=1, color='g', linewidth=1.0, animated=True)

        # create canvas
        self.figure_canvas = FigureCanvasTkAgg(self.figure, self)
        self.cid = self.figure_canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.figure_canvas.get_tk_widget().grid(column=0, row=2, sticky=tk.W, padx=5, pady=5, columnspan=5)

        # draw canvas (with only constant artists)
        self.figure_canvas.draw()
        plt.pause(0.5)

        # create background variable which will store a copy of already drawn artists on canvas
        self.bg = self.figure_canvas.copy_from_bbox(self.figure.bbox)

        # draw animated vertical lines
        self.ax.draw_artist(self.mark_line)
        self.ax.draw_artist(self.anim_line)
        
        # blit canvas (store it in memory)
        self.figure_canvas.blit(self.figure.bbox)
        plt.pause(0.1)


    def redraw(self, figure: Figure, figure_canvas: FigureCanvasTkAgg, bg: Any, artists: list[Any], ax: plt.axes, pos: list[float]) -> None:
        '''Use blitting to redraw canvas'''

        # recreate background (along with plot of audio data)
        figure_canvas.restore_region(region=bg)
        
        # update x coordinates of animated artists using their current positions and draw them on background (plot)
        for i in range(len(artists)):
            artists[i].set_xdata(pos[i])
            ax.draw_artist(artists[i])
        
        # blit canvas (store in memory) and execute all changes
        figure_canvas.blit(figure.bbox)
        figure_canvas.flush_events()


    def update_plot(self, last_pos) -> None:
        '''Updating plot in tkinter loop - drawing vertical line indicating current position'''
        
        # current sound data position becomes last position
        self.sound.last_pos = last_pos

        # update plot while playing
        if not self.stop_event.is_set():
            if not self.pause_event.is_set():
                if self.queue and not self.queue.empty():
                    # get current time of replay which play thread stores in queue
                    self.sound.pos = self.queue.get()
                else:
                    self.sound.pos = self.sound.last_pos
                
                self.redraw(self.figure, self.figure_canvas, self.bg, [self.anim_line, self.mark_line], self.ax, [self.sound.pos, self.sound.marker_pos])
                self.label_time['text'] = f"Time: {self.sound.pos:.3f} sec."
            else:
                pass
            
            # continue update in a loop (every 20 milliseconds, i.e. 50 frames/sec.)
            self.after_id = self.after(20, lambda: self.update_plot(self.sound.pos))
        # update when not playing
        else:
            # update when paused
            if not self.pause_event.is_set():
                self.sound.pos = self.time_axis[0]
                self.sound.last_pos = self.time_axis[0]
                self.label_time['text'] = f"Time: {self.sound.pos:.3f} sec."
            
            self.redraw(self.figure, self.figure_canvas, self.bg, [self.anim_line, self.mark_line], self.ax, [self.sound.pos, self.sound.marker_pos])
            self.play_button['state'] = "normal"

        # check if play thread is active and make sure that stop event is set if it is not
        play_thread_active = self.play_thread.is_alive() if hasattr(self, 'play_thread') else False
        if not play_thread_active:
            self.stop_event.set()

    
    # ============ Auxiliary methods ===============

    def set_buttons_state(self, state: str):
        '''Change sound and current data related buttons' state'''

        self.play_button['state'] = state
        self.pause_button['state'] = state
        self.stop_button['state'] = state


    def on_close(self) -> None:
        '''
        Called when close icon clicked
        Safely kill the app
        '''

        # update events
        self.close_event.set()
        self.stop_event.set()
        self.pause_event.clear()

        # make sure that tkinter loop is stopped
        try:
            self.after_cancel(self.after_id)
        except (AttributeError, Exception) as e:
            print(f"Error: {e}. Quitting anyway.")

        time.sleep(0.1)
        self.quit()
        self.destroy()
        
        print(f"App closed")


