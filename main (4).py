from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from numpy.core.numeric import indices
from pyqtgraph import *
from pyqtgraph import PlotWidget, PlotItem
import pyqtgraph as pg
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import pathlib
import numpy as np
from pyqtgraph.Qt import _StringIO
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from matplotlib.figure import Figure
import pyqtgraph.exporters
import math 
from DSP4LAST2 import MplCanvas, Ui_MainWindow
import winsound
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
import numpy as np
import matplotlib.pyplot as plt
from sympy import S, symbols, printing
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import rc
from numpy.lib.function_base import average
from numpy.lib.polynomial import polyfit, polyval
import seaborn as sns; sns.set_theme()
import pyqtgraph as pg
from PIL import Image
import statistics
from functools import reduce


class MainWindow(QtWidgets.QMainWindow):
    the_equations = []
    The_error_value = []
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #______________________________IMPORTANT INSTANCE VARIABLES__________________________________________________________#
        self.timer1 = QtCore.QTimer()
        self.mapping_canvas = MplCanvas(None, width=10, height=10, dpi=100)
        self.data_lines = []
        self.Timer = [self.timer1]
        self.SIGNAL_X = [[], [], []]
        self.SIGNAL_Y = [[], [], []]
        self.GraphicsView=[self.ui.graphicsView,self.ui.canvas]#ALL GRAPHIGSVIEW TO USE THEM WIH JUST INDEX
        self.white=mkPen(color=(255,255, 255))#white
        self.Color1=mkPen(color=(255, 0, 0))#RED
        self.Color2=mkPen(color=(0, 0, 0),style=QtCore.Qt.DotLine,width=2)#black
        self.Color3=mkPen(color=(0, 0, 255))#BLUE
        self.Color4=mkPen(color=(255, 200, 200), style=QtCore.Qt.DotLine)#Dotted pale-red line
        self.Color5=mkPen(color=(200, 255, 200), style=QtCore.Qt.DotLine,width=2)#Dotted pale-green line
        self.Color6=mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine)## Dotted pale-blue line
        self.COLOR_Pen=[self.Color1,self.Color2,self.Color3,self.Color4,self.Color5,self.Color6]#STORE COLORS TO BE USED WITH INDEX
        self.Pen = [self.Color1, self.Color2, self.Color3]
        self.poly_arr=np.zeros(50)
        self.ui.fitting_chunknumber_comboBox.hide()
        self.ui.Chunk_number_label.hide()
        self.ui.number_of_chunks_label.hide()
        self.ui.CHUNKS_NUMBER.hide()
        self.x_values=[]
        self.y_values=[]
        self.notadded=0
        self.firstfit=0
        self.Chunks_Number=1
         #hiding the mapping part at startup----------------
        self.ui.Chunk_number_label.hide()
        self.ui.Mchunk_n_spin.hide()
        self.ui.tableWidget.hide()
        self.ui.X_AXIS_COMBO.hide()
        self.ui.Y_AXIS_COMBO.hide()
        self.ui.map_chunck_number_label.hide()
        self.ui.X_AXIS_label.hide()
        self.ui.Y_AXIS_label.hide()
        self.ui.Polynomial_Order_label.hide()
        self.ui.Overlapping_label.hide()
        self.ui.Mapping_Progress_label.hide()
        self.ui.MAP_POLYNOMIAL_ORDER.hide()
        self.ui.Moverlap_spin.hide()
        self.ui.START_MAPPING.hide()
        self.ui.line_2.hide()
        self.ui.progressBar.hide()
        self.ui.CANCEL.hide()
        #___________________________________________CONNECTING BUTTONS WITH THEIR FUNCTIONS_______________________________________#
        self.ui.OPEN.triggered.connect(lambda: self.load())
        self.ui.actionCLEAR.triggered.connect(lambda: self.clear())
        self.ui.MAP.clicked.connect(lambda: self.map_error_show())
        self.ui.X_AXIS_COMBO.activated.connect(lambda: self.x_axis_parameter_show())
        self.ui.Y_AXIS_COMBO.activated.connect(lambda: self.y_axis_parameter_show())
        self.ui.START_MAPPING.clicked.connect(lambda: self.make_map())
        self.ui.FITTNG_USING_COMBO.activated.connect(lambda: self.fiiting_style())
        self.ui.FIT_POLYNOMIAL_ORDER.valueChanged.connect(lambda: self.change_poly())
        self.ui.CHUNKS_NUMBER.valueChanged.connect(lambda: self.fitting())
        self.ui.CLIPPING_SLIDER.valueChanged.connect(lambda: self.portion())
        self.ui.CANCEL.clicked.connect(lambda: self.cancel())
        self.ui.fitting_chunknumber_comboBox.activated.connect(lambda: self.latex())
 
    #_____________________________________________BUTTTONS FUNCTIONS_______________________________________________________# 
    def cancel(self):
        # self.ui.tableWidget.clear()
        self.ui.progressBar.setValue(0)
        self.ui.CANCEL.hide()
        self.ui.START_MAPPING.show()

    def read_file(self, Channel_ID):#----------------->>BROWSE TO READ THE FILE<<
        path = QFileDialog.getOpenFileName()[0]
        if pathlib.Path(path).suffix == ".csv":
            self.data = np.genfromtxt(path, delimiter=',')
            self.x_values = list(self.data[:, 0])
            self.y_values = list(self.data[:, 1])
            self.xpart=self.x_values
            self.ypart=self.y_values

    def load(self):#------------------------>>LOAD THE SIGNAL AND PLOT IT<<
        self.read_file(0)
        self.notadded=1
        self.CLR_Pen = self.Pen[0]
        self.GraphicsView[0].plotItem.addLegend()
        self.data_lines.append(self.GraphicsView[0].plot(self.x_values, self.y_values, linestyle="-", pen=self.CLR_Pen))
        self.GraphicsView[0].plotItem.setLabel("bottom", text="Time (ms)")
        self.GraphicsView[0].plotItem.showGrid(True, True, alpha=1)
        self.GraphicsView[0].plotItem.setLimits(xMin=0, xMax=10, yMin=-20, yMax=20)
        self.IDX = 0
        
    def fiiting_style(self):
        if(self.ui.FITTNG_USING_COMBO.currentIndex()==0):
             self.ui.fitting_chunknumber_comboBox.hide()
             self.ui.Chunk_number_label.hide()
             self.ui.number_of_chunks_label.hide()
             self.ui.CHUNKS_NUMBER.hide()
             
        elif (self.ui.FITTNG_USING_COMBO.currentIndex()==1):
            self.ui.fitting_chunknumber_comboBox.show()
            self.ui.Chunk_number_label.show()
            self.ui.number_of_chunks_label.show()
            self.ui.CHUNKS_NUMBER.show()

    def fitting(self):
        if (self.notadded==0 ):
            self.show_popup1()
        self.clear()
        self.ui.fitting_chunknumber_comboBox.clear()
        self.CLR_Pen = self.Pen[0]
        self.data_lines.append(self.GraphicsView[0].plot(self.x_values, self.y_values, linestyle="-", pen=self.CLR_Pen))
        self.the_equations = []
        self.The_error_value = []
        if(self.ui.FITTNG_USING_COMBO.currentIndex()==0):
            self.Chunks_Number=1
        if (self.ui.FITTNG_USING_COMBO.currentIndex()==1):
            self.Chunks_Number=self.ui.CHUNKS_NUMBER.value()
        for i in range(self.Chunks_Number):
            self.ui.fitting_chunknumber_comboBox.addItem('Chunk '+str(i+1))
        self.ui.fitting_chunknumber_comboBox.addItem('All')
        #fitting starts here----------------------------------------
        self.chunck_size=int(len(self.xpart)/self.Chunks_Number)#get chunk size
        self.start=0
        self.end=0
        for i in range(self.Chunks_Number):#loop and fit all the chunks one by one
            self.start=self.end
            self.end+=self.chunck_size
            print(self.end)
            self.rangex = self.xpart[ self.start:self.end:1]# x points of the chunk
            self.rangey = self.ypart[ self.start:self.end:1]# y points ............
            self.p = self.poly_arr[i]# select the poly n from list of poly numbers of all chunks
            print(self.p)
            self.z = np.polyfit(self.rangex, self.rangey, self.p)#the equation of fitting....(x,y,poly number)
            self.f = np.poly1d(self.z)#function of y-fitted to substitute with the x on it------y=sin(x)
            self.GraphicsView[0].plot(np.full(shape=10, fill_value=(self.end) / 1000, dtype=np.int),
                                      np.arange(-1, 1, 0.2), pen=self.Color3)#for drawing vertical lines between chunks
          
            if(self.ui.FITTNG_USING_COMBO.currentIndex()==0):#if one chunk
                self.x_new =  np.linspace( self.x_values[0],  self.x_values[-1], 50)
                self.y_new = self.f(self.x_new)#the new y val after fitting

            elif (self.ui.FITTNG_USING_COMBO.currentIndex()==1):#if multiple chunks
                self.x_new = np.linspace( self.rangex[0],  self.rangex[-1], 50)
                self.y_new = self.f(self.x_new)
            #---------------------changing equation format to latex and print it with error and plot the fitting-----------------------
            self.xx = symbols("x")
            self.poly = sum(S("{:6.2f}".format(v)) * self.xx ** i for i, v in enumerate(self.z[::-1]))
            self.eq_latex = "chunk" + str(i + 1) + ": " + str(printing.latex(self.poly))
            self.y_fitted = polyval(self.z, self.rangex)
            self.The_error = "Error" + " : " + str(round(abs((statistics.stdev(self.y_values)-statistics.stdev(self.y_fitted))/statistics.stdev(self.y_values))*100,2)) + " % "
            self.The_error_value.append(self.The_error)
            self.the_equations.append(self.eq_latex)
            print(self.eq_latex)
            self.GraphicsView[0].plotItem.addLegend()
            self.data_lines.append(self.GraphicsView[0].plot(self.x_new, self.y_new, linestyle="--", pen=self.Color2))
            self.COLOR_Pen = self.Pen[2]
            print(self.z)
        self.latex()
            
    def change_poly(self):
        if (self.ui.fitting_chunknumber_comboBox.currentIndex()== self.Chunks_Number):
            for i in range(50):
                self.poly_arr[i]=self.ui.FIT_POLYNOMIAL_ORDER.value()
        else:
            self.poly_arr[self.ui.fitting_chunknumber_comboBox.currentIndex()]=self.ui.FIT_POLYNOMIAL_ORDER.value()

        self.ui.CLIPPING_SLIDER.setValue(100)
        self.xpart= self.x_values[:int(len(self.x_values)*self.ui.CLIPPING_SLIDER.value()/100)]
        self.ypart= self.y_values[:int(len(self.x_values)*self.ui.CLIPPING_SLIDER.value()/100)]    
        self.fitting()
                        
    def portion(self):#slider to cover part of the signal
        self.xpart= self.x_values[:int(len(self.x_values)*self.ui.CLIPPING_SLIDER.value()/100)]
        self.ypart= self.y_values[:int(len(self.x_values)*self.ui.CLIPPING_SLIDER.value()/100)]
        self.fitting()
        
    def latex(self):  # equation show
        self.ui.axes.clear()
        self.ui.axes.text(0.5, 0.8, r"${}$".format(self.eq_latex), color="red",horizontalalignment='center',
     verticalalignment='center')
        self.ui.axes.text(0.5, 0.5 , r"${}$".format(self.The_error), color="blue",horizontalalignment='center',
     verticalalignment='center')
        self.ui.canvas.draw()
        
    def show_popup1(self):
        n=self.ui.msg1.exec_()

    def make_map(self):
        # self.ui.tableWidget.clear()
        #modify-----------------------------------------------------------------------------
        self.overlap_array=np.arange(0,self.ui.Moverlap_spin.value(),1)#make an array from 0 to ovalap value
        self.polynomial_array=np.arange(0,self.ui.MAP_POLYNOMIAL_ORDER.value(),1)#make an array from 0 to poly value
        self.chunk_n_array=np.arange(0,self.ui.Mchunk_n_spin.value(),1)#make an array from 0 to chunk n value
        #-----------------------------------------------------------------------------------------------
        self.ui.START_MAPPING.hide()
        self.ui.CANCEL.show()
        #1---------------------------------------------------------------------------------------------------------
        if (self.ui.X_AXIS_COMBO.currentIndex()==0 and self.ui.Y_AXIS_COMBO.currentIndex()==0):#poly & overlap
            self.ui.Mchunk_n_spin.hide()
            self.ui.map_chunck_number_label.hide()
            self.error_array=[np.zeros(self.ui.MAP_POLYNOMIAL_ORDER.value())]*int(self.ui.Moverlap_spin.value())
            # self.ui.tableWidget.setColumnCount(self.ui.MAP_POLYNOMIAL_ORDER.value())#make number of col
            # self.ui.tableWidget.setRowCount(len(self.overlap_array))#make number of rows
            self.completed = 0      # % of progress bar
            step=(100/(self.ui.MAP_POLYNOMIAL_ORDER.value()*self.ui.Moverlap_spin.value()))# step of progress bar
            self.ui.progressBar.show()
            for j in range(self.ui.Moverlap_spin.value()):#loop for rows
                for i in range(self.ui.MAP_POLYNOMIAL_ORDER.value()):#loop for colums
                    # self.ui.tableWidget.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem('poly degree='+str(i+1)))
                    # self.ui.tableWidget.setVerticalHeaderItem(j, QtWidgets.QTableWidgetItem('overlap='+str(j+1)+'%'))
                    p = polyfit(self.x_values,self.y_values,int(i))#make the fitting process for each colum value(poly number)
                    f = polyval(p,self.x_values)#the y array after fitted
                    #modify-------------------------------------------------
                    error=abs((statistics.stdev(self.y_values)-statistics.stdev(f))/statistics.stdev(self.y_values))*100
                    self.error_array[j][i]=error
                    print(str(error))
                    # self.ui.tableWidget.setItem(j,i,QtWidgets.QTableWidgetItem(str(error)))#print avg(the error val) in table
                    self.completed += step #increase the progress bar 
                    self.ui.progressBar.setValue(self.completed)
                if(j==self.ui.Moverlap_spin.value()-1):
                    self.ui.progressBar.setValue(100)#set progress bar at 100% at final value
                    
                    for i in reversed(range(self.ui.mapping_layout.count())): #LOOP TO DELETE THE OLD WIDGET>>SPECTROGRAM<< WITH ITS ALL ITEMS
                        print(i)
                        if i ==2:
                            self.ui.mapping_layout.itemAt(i).widget().deleteLater() 
                        elif i ==1:
                            self.ui.mapping_layout.itemAt(i).widget().deleteLater()
                    # #(1)
                    # ax=sns.heatmap(self.error_array) 
                    # plt.show()
                    #(2)
                    self.mapping_canvas = MplCanvas(self.ui.centralwidget, width=20, height=20, dpi=100)
                    self.ui.mapping_layout.addWidget(self.mapping_canvas)
                    self.mapping_canvas.axes.imshow(self.error_array, cmap='jet', aspect='auto')
                    im = self.mapping_canvas.axes.imshow(self.error_array, cmap='jet', aspect='auto')
                    self.mapping_canvas.axes.set_xlabel("polynomial number")
                    self.mapping_canvas.axes.xaxis.set_label_coords(1.13, -0.035)
                    self.mapping_canvas.axes.set_ylabel("overlapping")
                    self.mapping_canvas.figure.colorbar(im,label= "Error %", orientation="vertical")
                    self.mapping_canvas.draw()        
        #2---------------------------------------------------------------------------------------------------------
        elif (self.ui.X_AXIS_COMBO.currentIndex()==0 and self.ui.Y_AXIS_COMBO.currentIndex()==1):#poly & chunks number
            self.ui.Moverlap_spin.hide()
            self.ui.Overlapping_label.hide()
            # self.ui.tableWidget.setColumnCount(self.ui.MAP_POLYNOMIAL_ORDER.value())
            # self.ui.tableWidget.setRowCount(self.ui.Mchunk_n_spin.value())
            self.error_array=[np.zeros(self.ui.MAP_POLYNOMIAL_ORDER.value())]*int(self.ui.Mchunk_n_spin.value())
            self.completed = 0
            step=(100/(self.ui.MAP_POLYNOMIAL_ORDER.value()*self.ui.Mchunk_n_spin.value()))
            self.ui.progressBar.show()
            for j in range(1,self.ui.Mchunk_n_spin.value()+1):
                self.start=0
                self.end=0
                error_array_list=[]
                for c in range(1,j):
                    self.chunck_size=int(len(self.x_values)/j)
                    self.start=self.end
                    self.end+=self.chunck_size
                    self.rangex = self.x_values[ self.start:self.end:1]
                    self.rangey = self.y_values[ self.start:self.end:1]
                    error_array=[]
                    for i in range(1,self.ui.MAP_POLYNOMIAL_ORDER.value()+1):
                        # self.ui.tableWidget.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem('poly degree='+str(i)))
                        # self.ui.tableWidget.setVerticalHeaderItem(j, QtWidgets.QTableWidgetItem('chunk='+str(j)))
                        p = polyfit(self.rangex,self.rangey,int(i))
                        f = polyval(p,self.rangex)
                        error=abs((statistics.stdev(self.rangey)-statistics.stdev(f))/statistics.stdev(self.rangey))*100
                        error_array.append(error)
                    error_array_list.append(error_array)
                list_of_chunk_parts=[]
                for part in error_array_list:
                    list_of_chunk_parts=list_of_chunk_parts+part
                #print(""+str(j)+str(list_of_chunk_parts))
                for Pn in range(0,j-1):
                    eachpoly_part=[]
                    for Pnn in range(0,j-1):   
                        eachpoly_part.append(list_of_chunk_parts[Pnn])
                    print(str(eachpoly_part))
                    # self.ui.tableWidget.setItem(j-1,i-1,QtWidgets.QTableWidgetItem(str(max(eachpoly_part))))
                    self.error_array[j-1][i-1]=max(eachpoly_part)
                    self.completed += step
                    self.ui.progressBar.setValue(self.completed)
                if(j==self.ui.Mchunk_n_spin.value()):
                    print(str(self.error_array))
                    self.ui.progressBar.setValue(100)
                    for i in reversed(range(self.ui.mapping_layout.count())): #LOOP TO DELETE THE OLD WIDGET>>SPECTROGRAM<< WITH ITS ALL ITEMS
                        print(i)
                        if i ==2:
                            self.ui.mapping_layout.itemAt(i).widget().deleteLater()
                            
                        elif i ==1:
                            self.ui.mapping_layout.itemAt(i).widget().deleteLater() 
                    # #(1)
                    # ax=sns.heatmap(self.error_array) 
                    # plt.show()
                    #(2)
                    self.mapping_canvas = MplCanvas(self.ui.centralwidget, width=20, height=20, dpi=100)
                    self.ui.mapping_layout.addWidget(self.mapping_canvas)
                    self.mapping_canvas.axes.imshow(self.error_array, cmap='jet', aspect='auto')
                    im = self.mapping_canvas.axes.imshow(self.error_array, cmap='jet', aspect='auto')
                    self.mapping_canvas.axes.set_xlabel("polynomial number")
                    self.mapping_canvas.axes.xaxis.set_label_coords(1.13, -0.035)
                    self.mapping_canvas.axes.set_ylabel("Chunks number")
                    self.mapping_canvas.figure.colorbar(im, label= "Error %", orientation="vertical")
                    self.mapping_canvas.draw()        
        #3------------------------------------------------------------------------------------------------------     
        elif (self.ui.X_AXIS_COMBO.currentIndex()==1 and self.ui.Y_AXIS_COMBO.currentIndex()==1):#overlap & chunks number
            self.ui.MAP_POLYNOMIAL_ORDER.hide()
            self.ui.Polynomial_Order_label.hide()
            self.error_array=[np.zeros(self.ui.Moverlap_spin.value()+1)]*int(self.ui.Mchunk_n_spin.value())
            # self.ui.tableWidget.setColumnCount(self.ui.Moverlap_spin.value())
            # self.ui.tableWidget.setRowCount(self.ui.Mchunk_n_spin.value())
            self.completed = 0
            step=(100/(self.ui.Moverlap_spin.value()*self.ui.Mchunk_n_spin.value()))
            self.ui.progressBar.show()
            for j in range(1,self.ui.Mchunk_n_spin.value()+1):
                if j==1:
                    p = polyfit(self.x_values,self.y_values,5)
                    f = polyval(p,self.x_values)
                    error=abs((statistics.stdev(self.y_values)-statistics.stdev(f))/statistics.stdev(self.y_values))*100
                    for i in range(self.ui.Moverlap_spin.value()+1):
                        self.error_array[0][i]=error  
                else: 
                    start=0
                    end=0
                    chunck_size=int(len(self.x_values)/j)
                    start=end
                    end+=chunck_size
                    self.rangex = self.x_values[start:end:1]# x points of the chunk
                    self.rangey = self.y_values[start:end:1]# y points ............
                    p = polyfit(self.rangex,self.rangey,5)
                    f = polyval(p,self.rangex)
                    error_part1=abs((statistics.stdev(self.rangey)-statistics.stdev(f))/statistics.stdev(self.rangey))*100
                    part_at_different_overlaps=[np.zeros(self.ui.Moverlap_spin.value())]*int(j-1)
                    endpart=end
                    for part in range(2,j):
                        start=end
                        end+=chunck_size
                        for i in range(self.ui.Moverlap_spin.value()+1):
                            if start==end:
                                break
                            start=start-int((i*chunck_size)/100)
                            self.rangex = self.x_values[ start:end:1]# x points of the chunk
                            self.rangey = self.y_values[ start:end:1]# y points ............
                            p = polyfit(self.rangex,self.rangey,5)
                            f = polyval(p,self.rangex)
                            error=abs((statistics.stdev(self.rangey)-statistics.stdev(f))/statistics.stdev(self.rangey))*100
                            part_at_different_overlaps[int(part-2)][i-1]=error
                        endpart+=chunck_size
                        for i in range(self.ui.Moverlap_spin.value()+1):
                            chunks_error_array=[]
                            chunks_error_array.append(error_part1)
                            for part_1 in range(2,j):
                                chunks_error_array.append(part_at_different_overlaps[part_1-2][i-1])
                            self.error_array[j-1][i]=max(chunks_error_array)
            
                if(j==self.ui.Mchunk_n_spin.value()):
                    print(str(self.error_array))
                    self.ui.progressBar.setValue(100)
                    for i in reversed(range(self.ui.mapping_layout.count())): #LOOP TO DELETE THE OLD WIDGET>>SPECTROGRAM<< WITH ITS ALL ITEMS
                        print(i)
                        if i ==2:
                            self.ui.mapping_layout.itemAt(i).widget().deleteLater()
                        elif i ==1:
                            self.ui.mapping_layout.itemAt(i).widget().deleteLater() 
                    # #(1)
                    # ax=sns.heatmap(self.error_array) 
                    # plt.show()
                    #(2)
                    self.mapping_canvas = MplCanvas(self.ui.centralwidget, width=20, height=20, dpi=100)
                    self.ui.mapping_layout.addWidget(self.mapping_canvas)
                    self.mapping_canvas.axes.imshow(self.error_array, cmap='jet', aspect='auto')
                    im = self.mapping_canvas.axes.imshow(self.error_array, cmap='jet', aspect='auto')
                    self.mapping_canvas.axes.set_xlabel("overlapping")
                    self.mapping_canvas.axes.xaxis.set_label_coords(1.07, -0.035)
                    self.mapping_canvas.axes.set_ylabel("Chunks number")
                    self.mapping_canvas.figure.colorbar(im,label= "Error %", orientation="vertical")
                    self.mapping_canvas.draw()        

    def x_axis_parameter_show(self):
        if (self.ui.X_AXIS_COMBO.currentIndex()==0 ):
            self.ui.MAP_POLYNOMIAL_ORDER.show()
            self.ui.Polynomial_Order_label.show()
            
        elif(self.ui.X_AXIS_COMBO.currentIndex()==1 ):
            self.ui.Moverlap_spin.show()
            self.ui.Overlapping_label.show()
                
    def y_axis_parameter_show(self):
        if(self.ui.Y_AXIS_COMBO.currentIndex()==0 ):
            self.ui.Moverlap_spin.show()
            self.ui.Overlapping_label.show()
        elif(self.ui.Y_AXIS_COMBO.currentIndex()==1 ):
            self.ui.Mchunk_n_spin.show()
            self.ui.map_chunck_number_label.show()

    def map_error_show(self):
        self.ui.X_AXIS_COMBO.show()
        self.ui.Y_AXIS_COMBO.show()
        self.ui.X_AXIS_label.show()
        self.ui.Y_AXIS_label.show()
        self.ui.tableWidget.show()
        self.ui.START_MAPPING.show()
        self.ui.line_2.show()
  
    def clear(self):#------------------------------>>CLEAR THE 2 GRAPHS<<
        self.GraphicsView[0].clear()

    def plotting(self,GRAPHICSINDEX,X_ARRAY,Y_ARRAY,COLORLIST):#................>FUNCTION OF PLOTTING <FOR REDUCING THE REPEATATION OF CODE>
        self.GraphicsView[GRAPHICSINDEX].plot(X_ARRAY, Y_ARRAY, pen=COLORLIST)
        self.GraphicsView[GRAPHICSINDEX].plotItem.setLabel("bottom", text="Time (ms)")
        self.GraphicsView[GRAPHICSINDEX].plotItem.showGrid(True, True, alpha=1)
        self.GraphicsView[GRAPHICSINDEX].plotItem.setLimits(xMin=0, xMax=10, yMin=-20, yMax=20)

    def close_app(self):
        sys.exit()
#---------------------------------END OF MAINWINDOW CLASS---------------------------------------------#


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())