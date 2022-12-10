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
matplotlib.use('Qt5Agg')


plt.ion()
plt.style.use('dark_background')
fig = plt.figure()
spec = fig.add_gridspec(50, 50)
ax1 = fig.add_subplot(spec[0:22, 1:48])
ax2 = ax1.twinx()
ax3 = fig.add_subplot(spec[28:, 1:48])


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
        self.l_pass = '2';self.l_change1 = '0.05';self.l_change2 = '0.05';
        self.l_change3 = '0.05';self.l_change4 = '0.05';self.l_change5 = '0.05';
        self.l_change6 = '0.05';self.l_change7 = '0.05';self.l_change8 = '0.05';
        self.w_change2 = '60';self.w_change6 = '60';self.Cp = '5';
        self.w_pass = '12000';self.Rset='24'
        # STARTING VALUES
        self.rewards = []


    def generate_output_dim(self):
        # generator of possible values for the given circuit
        LDO_SIM.get_current_var_state(self)
        self.ch_value = 0.05 # um minimum technology value
        # FOR LOW CURRENT CMOS DEVICES
        self.dim_range_min = self.ch_value
        self.dim_range_max = 25000.
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
        for i in range(0,len(values)):
            if float(values[i]) <= float(self.ch_value):
                values[i] = self.ch_value
                reward -=1
            if float(values[i]) >= float(self.dim_range_max):
                values[i] = self.dim_range_max
                reward -=1
            if i == 8: # wpass
                if float(values[i])<= 10000.:
                    values[i] = 10000.
                    reward -= 1
                else:
                    pass
            else:
                if values[i] <=30 and i < 8:
                    reward +=1
                else:
                    reward -=1
                if values[i] <= 5 and i > 8:
                    reward += 1
                else:
                    reward -=1

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



    def play_step(self,plot_option,scores,mean_scores):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.grid()
        # ax3.grid()
        plt.tight_layout()
        file_name = '\Ldo_topology_sim'
        filepath = 'C:\PRACA\Omni-chip\PMU\LDO_new' + file_name
        LTC = SimCommander(filepath + '.asc')
        # idx = len(LTC.netlist) - 1
        # LTC.netlist.insert(idx,".tran 0 200u 0 1n")

        LTC.set_component_value('R4', '{RX}')
        LTC.set_component_value('R5', '{Rset}')

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
            ".tran 0 100u 0 10n",
            ".options temp 26.85",
            ".options plotwinsize 128",
            #".step temp -40 125 100", # temperature sweep
            ".step param RX 1 100 50")

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

        expected_no_points = 200
        for i in range(l.case_count):  # Iteration in simulation cases
            ttime = l.get_time(i)
            tz = expected_no_points / len(ttime)
            t = interpolation.zoom(ttime ,tz,mode='nearest')
            V_source = l.get_data('V(LDO_Out)', i)
            I_source = l.get_data('I(R4)', i)
            Vz = expected_no_points / len(V_source)
            Iz = expected_no_points / len(I_source)
            Vs = interpolation.zoom(V_source ,Vz,mode='nearest')
            Is = interpolation.zoom(I_source, Iz,mode='nearest')
            self.V_source_list.append(Vs)
            self.I_source_list.append(Is)
            if plot_option == 1:
                axplot1, = ax1.plot(t * 1. * 10 ** 6, Vs,'r')
                axplot2, = ax2.plot(t * 1. * 10 ** 6, Is, 'b--')

        LDO_SIM.get_current_var_state(self)
        self.current_state_var = np.array(self.state_var_str).astype(np.float)
        self.sim_state = self.current_state_var
        max_v,max_v_old = 0.,0.
        min_v,min_v_old = 0.,0.
        for i in range(0,len(self.V_source_list)):
            self.sim_state = np.concatenate([self.sim_state,self.V_source_list[i]])

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

            axplot3, = ax3.plot(scores,'white')
            axplot4, = ax3.plot(mean_scores, 'g')
            ax3.set_xlabel('No. Steps')
            ax3.set_ylabel('Reward')
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
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.1)
        os.remove(filepath + '_1.raw')
        os.remove(filepath + '_1.log')
        os.remove(filepath + '_1.op.raw')
        os.remove(filepath + '_1.net')
