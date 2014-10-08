import matplotlib.pyplot as plt
import numpy as np

import aurespf.solvers as au
from europe_plusgrid import europe_plus_Nodes
from FCResult import FCResult
from FCResult import myhist
from FlowCalculation import FlowCalculation

#### Colorschemes ################################

dth = (3.425)
dcolwidth = (2*3.425+0.236)

blue = '#134b7c'
yellow = '#f8ca00'
orange = '#e97f02'
brown = '#876310'
green = '#4a8e05'
red = '#ae1215'
purple = '#4f0a3d'
darkred= '#4f1215'
pink = '#bd157d'
lightpink = '#d89bc2'
aqua = '#37a688'
darkblue = '#09233b'
lightblue = '#8dc1e0'
grayblue = '#4a7fa2'

blue_cycle = [darkblue, blue, grayblue, lightblue]

color_cycle = [blue, red, orange, purple, green, \
               pink, lightblue, darkred, yellow,
               darkblue, grayblue, brown]

plt.rc('lines', lw=2)
plt.rcParams['axes.color_cycle'] = color_cycle

all_links = ['AUT to CHE',
             'AUT to CZE',
             'AUT to HUN',
             'AUT to DEU',
             'AUT to ITA',
             'AUT to SVN',
             'FIN to SWE',
             'FIN to EST',
             'NLD to NOR',
             'NLD to BEL',
             'NLD to GBR',
             'NLD to DEU',
             'BIH to HRV',
             'BIH to SRB',
             'FRA to BEL',
             'FRA to GBR',
             'FRA to CHE',
             'FRA to DEU',
             'FRA to ITA',
             'FRA to ESP',
             'NOR to SWE',
             'NOR to DNK',
             'GBR to IRL',
             'POL to CZE',
             'POL to DEU',
             'POL to SWE',
             'POL to SVK',
             'BGR to GRC',
             'BGR to ROU',
             'BGR to SRB',
             'GRC to ITA',
             'PRT to ESP',
             'CHE to DEU',
             'CHE to ITA',
             'HRV to HUN',
             'HRV to SRB',
             'HRV to SVN',
             'ROU to HUN',
             'ROU to SRB',
             'CZE to DEU',
             'CZE to SVK',
             'HUN to SRB',
             'HUN to SVK',
             'DEU to SWE',
             'DEU to DNK',
             'DEU to LUX',
             'SWE to DNK',
             'ITA to SVN',
             'EST to LVA',
             'LVA to LTU']


def get_data(filename, field, path='./results/'):
    """ Returns the data in a certain field,
        from an FCResult file.
        See the FCResult class for available fields.

        Example
        -------
        >>> get_data("eurasia_aHE_0.95q99_lin.pkl", 'Total_TC')

        """

    result = FCResult(filename, path=path)
    returnvalue = result.cache[0][field]

    return returnvalue

def get_flow_hist_data(filename, link_number, path='./results/LinkCapSweeps/', bins=530, normed=False):
    F = np.load(path+filename)
    value, count = myhist(F[link_number], bins=bins, normed=normed)
    return value, count


def make_all_solvermode_flowhists():
    lin_modes = ['lin', 'DC_lin']
    sqr_modes = ['sqr', 'DC_sqr']

    capacities = [str(a) + 'q99' for a in np.linspace(0.05, 1.5, 30)]
    capacities.append('copper')

    for link in range(50):
        for c in capacities:
            lin_fclist = [FlowCalculation('Europe', 'aHE', c, m)\
                                                     for m in lin_modes]
            sqr_fclist = [FlowCalculation('Europe', 'aHE', c, m)\
                                                     for m in sqr_modes]

            plot_flow_hists(lin_fclist, link, interactive=False, semilogy=True)
            plot_flow_hists(sqr_fclist, link, interactive=False, semilogy=True)

    return



def plot_flow_hists(fclist, link_number, varparam='solvermode',\
                                interactive=True, figfilename=None, \
                                savepath='./results/figures/FlowHists/',
                                semilogy=False):

    plt.close()
    if interactive:
        plt.ion()

    for fc in fclist:
        filename = str(fc) + '_flows.npy'
        value, count = get_flow_hist_data(filename, link_number)
        if varparam=='solvermode':
            label = fc.pretty_solvermode()
        if semilogy:
            plt.semilogy(value, count, label=label)
        else:
            plt.plot(value, count, label=label)

    plt.legend()
    plt.title(all_links[link_number])
    plt.xlabel('Power flow [MW]')
    plt.ylabel('Count')

    if not figfilename:
        figfilename = ''.join([all_links[link_number].replace(' ', ''), '_',\
                                fclist[0].solvermode, '_', \
                                fclist[0].capacities, '.pdf'])
    if not interactive:
        plt.savefig(savepath+figfilename)
        plt.close()


def make_all_bal_vs_trans():
    modes = ['lin', 'sqr']
    ydatalabels = ['BE', 'BC']
    for m in modes:
        for y in ydatalabels:
            plot_bal_vs_trans(ydatalabel=y, mode=m, interactive=False)

    return


def plot_bal_vs_trans(ydatalabel='BE', mode='lin', interactive=True,
        figfilename=None, savepath='./results/figures/BalvsTrans/'):

    datapath = './results/LinkCapSweeps/'
    plt.close()
    if interactive:
        plt.ion()

    scalefactors = np.linspace(0,1.5,31)
    copperTC = sum(np.load('./results/Europe_copper_linkcaps_'+mode+'.npy'))
    TC = copperTC*scalefactors/1e6
    total_mean_load = sum(np.load('./results/Europe_meanloads.npy'))
    modes = [mode, 'DC_' + mode]
    for m in modes:
        ydata = []
        for a in scalefactors:
            capstr = str(a) + 'q99'
            fc = FlowCalculation('Europe', 'aHE', capstr, m)
            filename = str(fc)+'.pkl'
            ydata.append(sum(get_data(filename, ydatalabel, datapath))\
                            /total_mean_load)
        plt.plot(TC, ydata,\
                         label=fc.pretty_solvermode()) # TC in TW

    plt.ylim(0,1.1*max(ydata))
    plt.xlim(0,max(TC))
    plt.legend(loc=4)
    plt.xlabel('Total transmission capacity [TW]')
    if ydatalabel=='BE':
        plt.ylabel('Backup energy [normalized]')
    elif ydatalabel=='BC':
        plt.ylabel('Backup capacity [normalized]')

    if not figfilename:
        figfilename = ydatalabel + 'vsTC_' + mode + '.pdf'
    if not interactive:
        plt.savefig(savepath+figfilename)
        plt.close()

def plot_FDC_vs_oldF(capacities, mode, link_number, interactive=True, path='./results/LinkCapSweeps/'):

    if interactive:
        plt.ion()

    FDC = np.load(''.join([path, 'Europe_aHE_', capacities, '_DC_',\
                           mode, '_flows.npy']))[link_number]

    Fold = np.load(''.join([path, 'Europe_aHE_', capacities, '_',\
                           mode, '_flows.npy']))[link_number]

    plt.plot(Fold, FDC, '.')
    print np.corrcoef(FDC, Fold)


def almost_equal(x, y, abstol=0.01):
    return (np.abs(x-y)<=abstol)


def explain_peak_scatter_plot(filename='Europe_aHE_0.1q99_DC_sqr_flows.npy',\
                              peak_value=394.476, links=range(6),\
                              link_with_peak=0):

    plt.close()
    plt.ion()

    F = np.load('./results/LinkCapSweeps/'+filename)
    t = np.arange(len(F[0]))

    for l in links:
        plt.plot(t[almost_equal(F[link_with_peak], peak_value)],\
                 F[l][almost_equal(F[link_with_peak], peak_value)], '.',\
                 label=all_links[l])

        plt.legend()
    plt.xlabel('t [hour]')
    plt.ylabel(r'$F_l$' + ' [MW]')


