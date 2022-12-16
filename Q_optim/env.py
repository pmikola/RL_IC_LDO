import ltspice
import matplotlib.pyplot as plt
import numpy as np
import os
from PyLTSpice.LTSpice_RawWrite import Trace, LTSpiceRawWrite
from PyLTSpice.LTSpiceBatch import SimCommander
from matplotlib.ticker import FormatStrFormatter
from PyLTSpice.LTSpice_RawRead import LTSpiceRawRead
import time
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation
from scipy.fft import fft, fftfreq, fftshift
import pywt
from pywt import wavedec
matplotlib.use('Qt5Agg')


plt.ion()
plt.style.use('dark_background')
fig = plt.figure(figsize=(9, 6), dpi=100) # dpi=pixelc

spec = fig.add_gridspec(50, 100)
ax1 = fig.add_subplot(spec[0:22, 1:38])
ax2 = ax1.twinx()
ax3 = fig.add_subplot(spec[28:, 1:38])
ax4 = ax3.twinx()
ax5 = fig.add_subplot(spec[0:22, 62:99])
ax5i = ax5.twinx()
ax6 = fig.add_subplot(spec[28:, 62:99])
ax7 = ax6.twinx()

def decimalToBinary(n, x):
    if n > 1:
        x = decimalToBinary(n // 2, x)
    return x + 1

class LDO_SIM:
    def __init__(self):
        super().__init__()
        # STARTING VALUES
        self.w_change1 = '1';self.w_change3 = '15';self.w_change4 = '15';
        self.w_change5 = '1';self.w_change7 = '10';self.w_change8 = '10';
        self.l_pass = '0.75';self.l_change1 = '0.05';self.l_change2 = '0.05';
        self.l_change3 = '0.05';self.l_change4 = '0.05';self.l_change5 = '0.05';
        self.l_change6 = '0.05';self.l_change7 = '0.05';self.l_change8 = '0.05';
        self.w_change2 = '60';self.w_change6 = '60';self.Cp = '5';
        self.w_pass = '35000';self.Rset='24'
        # STARTING VALUES
        self.rewards = []


    def generate_output_dim(self):
        # generator of possible values for the given circuit
        LDO_SIM.get_current_var_state(self)
        self.ch_value = 0.05 # um minimum technology value
        # FOR LOW CURRENT CMOS DEVICES
        self.dim_range_min = self.ch_value
        self.dim_range_max = 55000.
        #self.nbits_multiplier=0
        #self.nbits_multiplier = decimalToBinary(int(self.hnum), 0)
        self.num_of_var = 20 # output values n heads number
        #self.nbits_dev = 0
        #self.nbits_dev = decimalToBinary(int(self.num_of_dev), 0)
        self.stop_bit = 0
        #print("Net output length : ", "No bits for Multiply : ",nbits_multiplier,"No. of devices : ",nbits_var,"stop : ",stop_bit, "||",nbits_multiplier+nbits_var+stop_bit)
        self.output_shape = np.zeros((self.num_of_var),dtype=np.float)



    def get_current_var_state(self):
        self.state_var_str = []
        self.state_var_str.append(self.w_change1)
        self.state_var_str.append(self.w_change2)
        self.state_var_str.append(self.w_change3)
        self.state_var_str.append(self.w_change4)
        self.state_var_str.append(self.w_change5)
        self.state_var_str.append(self.w_change6)
        self.state_var_str.append(self.w_change7)
        self.state_var_str.append(self.w_change8)
        self.state_var_str.append(self.w_pass)
        self.state_var_str.append(self.l_change1)
        self.state_var_str.append(self.l_change2)
        self.state_var_str.append(self.l_change3)
        self.state_var_str.append(self.l_change4)
        self.state_var_str.append(self.l_change5)
        self.state_var_str.append(self.l_change6)
        self.state_var_str.append(self.l_change7)
        self.state_var_str.append(self.l_change8)
        self.state_var_str.append(self.l_pass)
        self.state_var_str.append(self.Cp)
        self.state_var_str.append(self.Rset)

    def set_current_var(self,values,reward):

        self.w_change1 = str(values[0])
        self.w_change2 = str(values[1])
        self.w_change3 = str(values[2])
        self.w_change4 = str(values[3])
        self.w_change5 = str(values[4])
        self.w_change6 = str(values[5])
        self.w_change7 = str(values[6])
        self.w_change8 = str(values[7])
        self.w_pass = str(values[8])
        self.l_change1 = str(values[9])
        self.l_change2 = str(values[10])
        self.l_change3 = str(values[11])
        self.l_change4 = str(values[12])
        self.l_change5 = str(values[13])
        self.l_change6 = str(values[14])
        self.l_change7 = str(values[15])
        self.l_change8 = str(values[16])
        self.l_pass = str(values[17])
        self.Cp = str(values[18])
        self.Rset = str(values[19])
        return reward



    def play_step(self,plot_option,scores,mean_scores,loss_list):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax5i.clear()
        ax6.clear()
        ax7.clear()
        ax1.grid()
        # ax5.grid()
        # ax2.grid()
        # ax3.grid()
        # ax3.grid()
        file_name = '\Ldo_topology_sim'
        filepath = 'C:\PRACA\Omni-chip\PMU\LDO_new' + file_name
        LTC = SimCommander(filepath + '.asc')
        # idx = len(LTC.netlist) - 1
        # LTC.netlist.insert(idx,".tran 0 200u 0 1n")

        LTC.set_component_value('R4', '1Meg')
        LTC.set_component_value('R5', '{Rset}')
        LTC.set_element_model("I1", "PWL(0us 100uA 20us 1000uA 30us 10mA 50us 50mA 100us 100mA 150us 200mA 190us 350mA)")
        LTC.add_instructions(
            # PARAMS
            ".param Wpass=" + self.w_pass + "u" + " Lpass="+self.l_pass+"u",
            ".param l1=" + self.l_change1 + "u" + " w1=" + self.w_change1 + "u",
            ".param l2=" + self.l_change2 + "u" + " w2=" + self.w_change2 + "u",
            ".param l3=" + self.l_change3 + "u" + " w3=" + self.w_change3 + "u",
            ".param l4=" + self.l_change4 + "u" + " w4=" + self.w_change4 + "u",
            ".param l5=" + self.l_change5 + "u" + " w5=" + self.w_change5 + "u",
            ".param l6=" + self.l_change6 + "u" + " w6=" + self.w_change6 + "u",
            ".param l7=" + self.l_change7 + "u" + " w7=" + self.w_change7 + "u",
            ".param l8=" + self.l_change8 + "u" + " w8=" + self.w_change8 + "u",
            ".param Cm=" + self.Cp + "p",".param Rset=" + self.Rset + "k"
        )

        LTC.add_instructions(
            # STATIC PARAMS
            ".include cmosedu_models.txt",
            ".tran 0 200u 0 5n",
            ".options temp 26.85",
            ".options plotwinsize 128"
            #".step temp -40 125 100", # temperature sweep
            #".step param RX 1 100 50"
        )

        LTC.run()
        # print(LTC.failSim)
        LTC.wait_completion()
        print('Successful/Total Simulations: ' + str(LTC.okSim) + '/' + str(LTC.runno))

        ####### PRINT RESULTS ######
        l = ltspice.Ltspice(filepath + '_1.raw')
        l.parse()  # Data loading sequence. It may take few minutes for huge file.
        vars = l.variables
        ttime = l.get_time()


        self.V_source_list = []
        self.I_source_list = []
        self.V_spectrum_list = []
        self.V_spec_fftfreq_list = []
        self.Vs_fft = []
        self.cwt_list = []
        self.dwt_list = []
        # number of signal points
        no_points = 200
        # sample spacing
        T = 1. / 10*no_points
        # cwt no points
        t = np.linspace(-1, 1, no_points, endpoint=False)
        # wavelets width
        widths = np.arange(1, 31)
        cwt_wavelet = 'mexh'
        dwt_wavelet = 'db1'

        for i in range(l.case_count):  # Iteration in simulation cases
            ttime = l.get_time(i)
            tz = no_points / len(ttime)
            t = interpolation.zoom(ttime ,tz,mode='nearest')
            V_source = l.get_data('V(LDO_Out)', i)
            I_source = l.get_data('I(I1)', i)
            Vz = no_points / len(V_source)
            Iz = no_points / len(I_source)
            Vs = interpolation.zoom(V_source ,Vz,mode='nearest')
            Is = interpolation.zoom(I_source, Iz,mode='nearest')
            self.V_source_list.append(Vs)
            self.I_source_list.append(Is)
            self.Vs_fft.append(fftshift(fft(Vs)))
            self.V_spec_fftfreq_list.append(fftshift(fftfreq(no_points, T)))
            self.V_spectrum_list.append(1.0 / no_points * np.abs(fftshift(fft(Vs))))
            cwtmatr, freqs = pywt.cwt(V_source, widths, cwt_wavelet)
            self.cwt_temp = []
            self.dwt_temp = []
            for j in range(0,30):
                #print(int(cwtmatr[j].size))
                cwtmatr_scale = no_points / int(cwtmatr[j].size)
                self.cwt_temp.append(interpolation.zoom(cwtmatr[j], cwtmatr_scale, mode='nearest'))

            self.cwt_list.append(self.cwt_temp)
            dwt_coeffs = wavedec(self.V_source_list[i], dwt_wavelet, level=2)
            for j in range(0,int(np.array(dwt_coeffs).size)):
                #print(int(cwtmatr[j].size))
                dwt_scale = no_points / int(dwt_coeffs[j].size)
                self.dwt_temp.append(interpolation.zoom(dwt_coeffs[j], dwt_scale, mode='nearest'))

            self.dwt_list.append(self.dwt_temp)
            #print(self.dwt_list[i])

            if plot_option == 1:
                axplot1, = ax1.plot(t * 1. * 10 ** 6, Vs,'r')
                axplot2, = ax2.plot(t * 1. * 10 ** 6, Is, 'b--')
                axplot6r, = ax5.plot(self.V_spec_fftfreq_list[i],self.V_spectrum_list[i],'m')
                axplot6i, = ax5i.plot(self.V_spec_fftfreq_list[i],self.Vs_fft[i].imag,'c--')
                axplot7, = ax6.plot(self.cwt_list[i][0],'c')
                axplot8, = ax6.plot(self.cwt_list[i][15],'m')
                axplot9, = ax6.plot(self.cwt_list[i][29],'y')
                axplot10, = ax7.plot(self.dwt_list[i][0], 'c-.')
                axplot11, = ax7.plot(self.dwt_list[i][1], 'm-.')
                axplot12, = ax7.plot(self.dwt_list[i][2], 'y-.')
                # print(self.Vs_fft[i].imag)
                # print(len(self.cwt_list[i][15]))


        LDO_SIM.get_current_var_state(self)
        self.current_state_var = np.array(self.state_var_str).astype(np.float)
        self.sim_state = self.current_state_var
        for i in range(0,len(self.V_source_list)):
            self.sim_state = np.concatenate([self.sim_state,self.V_source_list[i]])
            self.sim_state = np.concatenate([self.sim_state,self.cwt_list[i][0]])
            self.sim_state = np.concatenate([self.sim_state, self.cwt_list[i][15]])
            self.sim_state = np.concatenate([self.sim_state, self.cwt_list[i][29]])
            self.sim_state = np.concatenate([self.sim_state, self.dwt_list[i][0]])
            self.sim_state = np.concatenate([self.sim_state, self.dwt_list[i][1]])
            self.sim_state = np.concatenate([self.sim_state, self.dwt_list[i][2]])
            self.sim_state = np.concatenate([self.sim_state, self.V_spectrum_list[i]])
            self.sim_state = np.concatenate([self.sim_state, self.Vs_fft[i].imag])


        if plot_option == 1:
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            ax1.tick_params(axis='y', colors='red')
            ax2.tick_params(axis='y', colors='blue')
            ax1.yaxis.label.set_color('red')
            ax2.yaxis.label.set_color('blue')
            ax1.set_xlabel('Time [us]')
            ax1.set_ylabel('Voltage [V]')
            ax2.set_ylabel('Current [A]')
            ax2.set_yscale('log')

            axplot3, = ax3.plot(scores,'white')
            axplot4, = ax3.plot(mean_scores, 'g')
            ax3.set_xlabel('No. Steps')
            ax3.set_ylabel('Reward')
            # ax3.set_yscale('log')

            axplot3.set_ydata(scores)#
            axplot4.set_ydata(mean_scores)
            if len(self.V_source_list) > 1:
                axplot1.set_ydata(self.V_source_list[-2])
                axplot2.set_ydata(self.I_source_list[-2])

            if len(self.V_source_list) > 2:
                axplot1.set_ydata(self.V_source_list[-3])
                axplot2.set_ydata(self.I_source_list[-3])


            axplot1.set_ydata(self.V_source_list[-1])
            axplot2.set_ydata(self.I_source_list[-1])

            axplot5, = ax4.plot(loss_list[-len(mean_scores)-1:-1], 'orange',linestyle=':')
            axplot5.set_ydata(loss_list[-len(mean_scores)-1:-1])
            ax4.tick_params(axis='y', colors='orange')
            ax4.yaxis.label.set_color('orange')
            ax4.set_ylabel('Loss')
            ax4.set_yscale('log')

            ax5.set_xlabel('Freq components')
            ax5.set_ylabel('Real')
            ax5i.set_ylabel('Imag')
            ax5.tick_params(axis='y', colors='m')
            ax5i.tick_params(axis='y', colors='c')
            ax5.yaxis.label.set_color('m')
            ax5i.yaxis.label.set_color('c')
            axplot6r.set_ydata(self.V_spectrum_list[-1])
            axplot6i.set_ydata(self.Vs_fft[-1].imag)

            ax6.set_ylabel('CWT Coeffs -')
            ax6.set_xlabel('Time [us] / Freq')
            axplot7.set_ydata(self.cwt_list[-1][0])
            axplot8.set_ydata(self.cwt_list[-1][15])
            axplot9.set_ydata(self.cwt_list[-1][29])
            axplot10.set_ydata(self.dwt_list[-1][0])

            ax7.set_ylabel('DWT Coeffs -.')
            axplot10.set_ydata(self.dwt_list[-1][0])
            axplot11.set_ydata(self.dwt_list[-1][1])
            axplot12.set_ydata(self.dwt_list[-1][2])

            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.1)
        os.remove(filepath + '_1.raw')
        os.remove(filepath + '_1.log')
        os.remove(filepath + '_1.op.raw')
        os.remove(filepath + '_1.net')
