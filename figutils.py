import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shapefile
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection
from scipy.optimize import curve_fit as fit
import matplotlib
import numpy as np
import networkx as nx
import copy


import aurespf.solvers as au
import PCA_tools as PCA
from europe_plusgrid import europe_plus_Nodes
from FCResult import FCResult
from FCResult import myhist
from FlowCalculation import FlowCalculation

#### Colorschemes ################################

dcolwidth = (2*3.425+0.236)
colwidth = (3.425)

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
darkgrey = '#8a7e7e'
black = '#000000'

blue_cycle = [darkblue, blue, grayblue, lightblue]

color_cycle = [blue, red, orange, purple, green, \
               pink, lightblue, darkred, yellow,
               darkblue, grayblue, brown]

plt.rc('lines', lw=2)
plt.rcParams['axes.color_cycle'] = color_cycle

#### Commmom variables ###################################
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

links_of_interest = ['AUT to DEU', 'BGR to ROU', 'FIN to SWE', 'POL to DEU',
                     'PRT to ESP', 'FRA to GBR', 'GRC to ITA']

all_countries = ['AUT', 'FIN', 'NLD', 'BIH', 'FRA', 'NOR', 'BEL','GBR', \
                 'POL', 'BGR', 'GRC', 'PRT', 'CHE', 'HRV', 'ROU', 'CZE',\
                 'HUN', 'SRB', 'DEU', 'IRL', 'SWE', 'DNK', 'ITA', 'SVN',\
                 'ESP', 'LUX', 'SVK', 'EST', 'LVA', 'LTU']

ISO3ISO2dict = {'AUT':'AT', 'FIN':'FI', 'NLD':'NL', 'BIH':'BA',\
            'FRA':'FR', 'NOR':'NO', 'BEL':'BE','GBR':'GB', 'POL':'PL',\
            'BGR':'BG', 'GRC':'GR', 'PRT':'PT', 'CHE':'CH', 'HRV':'HR',\
            'ROU':'RO', 'CZE':'CZ', 'HUN':'HU', 'SRB':'RS', 'DEU':'DE',\
            'IRL':'IE', 'SWE':'SE', 'DNK':'DK', 'ITA':'IT', 'SVN':'SI',\
            'ESP':'ES', 'LUX':'LU', 'SVK':'SK', 'EST':'EE', 'LVA':'LV',\
            'LTU':'LT'}

all_countries_ISO2 = [ISO3ISO2dict[c] for c in all_countries]
## translate all_links to ISO2 country codes
all_links_ISO2 = []
for l in all_links:
    l_ISO2 = copy.deepcopy(l)
    for c in all_countries:
        if c in l_ISO2:
            l_ISO2 = l_ISO2.replace(c, ISO3ISO2dict[c])
    all_links_ISO2.append(l_ISO2)


## Dictionary with index of the countries in the shapefiles
shapefile_index = {'AUT': 16,
                     'BEL': 19,
                     'BGR': 23,
                     'BIH': 26,
                     'CHE': 40,
                     'CZE': 60,
                     'DEU': 61,
                     'DNK': 64,
                     'ESP': 71,
                     'EST': 72,
                     'FIN': 74,
                     'FRA': 77,
                     'GBR': 81,
                     'GRC': 90,
                     'HRV': 99,
                     'HUN': 101,
                     'IRL': 107,
                     'ITA': 112,
                     'LTU': 136,
                     'LUX': 137,
                     'LVA': 138,
                     'NLD': 168,
                     'NOR': 169,
                     'POL': 182,
                     'PRT': 185,
                     'ROU': 190,
                     'SRB': 210,
                     'SVK': 213,
                     'SVN': 214,
                     'SWE': 215}


##############################################################################
############## Auxilary functions ############################################
##############################################################################

def get_pretty_mode(mode):
    """ Auxilary function that transforms 'lin' into 'Localized flow'
        and 'sqr' into 'Synchronized flow'.

        """

    return mode.replace('lin', 'Localized').replace('sqr', 'Synchronized')\
                        + ' flow'

def get_pretty_imp_and_mode(mode):
    prettymode = get_pretty_mode(mode)
    if 'DC' in mode:
        prettymodeimp = prettymode[3:-4] + 'phase angle flow'
    else:
        prettymodeimp = prettymode[0:-4] + 'minimal dissipation flow'

    return prettymodeimp

def get_beta_string(capacities_str):
    """ Auxilary function to get a beta value from the capacities field
        in a FlowCalculation object. Returns '0.45' if given '0.45q99' and
        r'$\infty$' if given 'copper'.

        """

    if capacities_str=='copper':
        return r'$\infty$'
    else:
        return capacities_str[0:-3]


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


def get_flow_hist_data(filename, link_number, path='./results/LinkCapSweeps/',\
                        bins=530, normed=False):
    """ This function returns values and count for a histogram along the
        flow on the specified link, with a specified filename.
        The default bins is 530 uniformly spaced bins -
        ~ sqrt(len(timeseries))

        """

    F = np.load(path+filename)
    value, count = myhist(F[link_number], bins=bins, normed=normed)
    return value, count


def almost_equal(x, y, abstol=0.01):
    """ This is a auxilary function returning True if the input values are
        closer than a given absolut tolerance (0.01).

        """

    return (np.abs(x-y)<=abstol)


def get_maxed_out_prob(F, link, reltol=1e-3):
    """ Returns the probability that the flow on a link is
        within reltol of the transmission capacity of the link.

        """

    fminus = min(F[link])
    fplus = max(F[link])
    lenght_of_timeseries = len(F[link])
    p_plus = float(len(F[link][np.where(F[link]>(1-reltol)*fplus)]))\
                    /lenght_of_timeseries
    p_minus = float(len(F[link][np.where(F[link]<(1-reltol)*fminus)]))\
                    /lenght_of_timeseries

    return p_plus+p_minus



############################################################################
########## Plotting functions ##############################################
############################################################################

def make_all_solvermode_flowhists():
    """ This function calls plot_flow_hists(...) for all the possible
        combinations (link, capacity, solvermode..)

        """

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


def make_flow_hist_gif_figs(mode, link_number, savepath, xlim, ylim):
    modes = [mode, 'DC_'+mode]
    caps = [str(a) + 'q99' for a in np.linspace(0.05, 1.5, 30)]
    caps.append('copper')
    caps.reverse()

    for c in caps:
        fclist = [FlowCalculation('Europe', 'aHE', c, m) for m in modes]
        figfilename = all_links_ISO2[link_number].replace(' ', '') + '_' + mode\
                        + ('_%02i.pdf'%(caps.index(c)))
        plot_flow_hists(fclist=fclist, link_number=link_number,\
                        interactive=False, figfilename=figfilename,\
                        savepath=savepath, varparam='implementation',
                        semilogy=True, xlim=xlim, ylim=ylim, showbeta=True)


def plot_flow_hists(fclist, link_number, varparam='solvermode',\
                                interactive=True, figfilename=None, \
                                savepath='./results/figures/FlowHists/',
                                semilogy=True, xlim=None, ylim=None,
                                showbeta=False):
    """ This function plots flow histograms for the configurations described
        in the FlowCalculation objects in fclist for the flow along the
        specified link.

        """

    plt.close()
    if interactive:
        plt.ion()

    for fc in fclist:
        filename = str(fc) + '_flows.npy'
        value, count = get_flow_hist_data(filename, link_number)
        if varparam=='solvermode':
            label = get_pretty_mode(fc.solvermode)
        if varparam=='implementation':
            if 'DC' in fc.solvermode:
                label = 'Phase angle flow'
            else:
                label = 'Minimal dissipation flow'
        if semilogy:
            plt.semilogy(value, count, label=label)
        else:
            plt.plot(value, count, label=label)

    plt.legend()
    if not showbeta:
        plt.title(all_links_ISO2[link_number])
    else:
        beta_str = get_beta_string(fclist[0].capacities)
        plt.title(all_links_ISO2[link_number] + ': ' + r'$\beta=$' + beta_str)

    plt.xlabel('Power flow [MW]')
    plt.ylabel('Count')

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    if not figfilename:
        figfilename = ''.join([all_links_ISO2[link_number].replace(' ', ''), '_',\
                                fclist[0].solvermode, '_', \
                                fclist[0].capacities, '.pdf'])
    if not interactive:
        plt.savefig(savepath+figfilename)
        plt.close()


def flow_hist_subplots(mode, link_numbers=[24, 31],\
                         capacities=['0.1q99', 'copper']):
    plt.close('all')
    subplot_count = 1
    fig = plt.figure(figsize=(10,9))

    for link_number in link_numbers:
        for caps in capacities:
            filename = 'Europe_aHE_' + caps + '_' + mode + '_flows.npy'
            DC_filename = 'Europe_aHE_' + caps + '_DC_' + mode + '_flows.npy'
            value, count = get_flow_hist_data(filename, link_number)
            DC_value, DC_count = get_flow_hist_data(DC_filename, link_number)
            plt.subplot(2,2,subplot_count)
            plt.semilogy(value/1e3, count, label='Minimal dissipation flow')
            plt.semilogy(DC_value/1e3, DC_count, label='Phase angle flow')
            beta_str = get_beta_string(caps)
            plt.title(all_links_ISO2[link_number] + ': ' \
                            + r'$\beta=$' + beta_str)
            plt.xlabel(r'$F_l$' + ' [GW]')
            plt.ylabel('Count')
            if subplot_count == 2 and mode=='lin':
                plt.xlim((-14, 21))
            if subplot_count == 3 and mode=='sqr':
                plt.xlim((-1.1, 1.1))
            plt.legend(prop={'size':10})
            subplot_count += 1

    fig.tight_layout()
    fig.subplots_adjust(top=0.915)
    fig.suptitle(get_pretty_mode(mode), fontsize=24)
    fig.savefig('./results/figures/four_flow_hists_' + mode + '.pdf')

def make_all_bal_vs_trans():
    """ Calls plot_bal_vs_trans(...) in the four relevant cases.

        """
    modes = ['lin', 'sqr']
    ydatalabels = ['BE', 'BC']
    for m in modes:
        for y in ydatalabels:
            plot_bal_vs_trans(ydatalabel=y, mode=m, interactive=False)

    return


def plot_bal_vs_trans_quotients(interactive=True):
    datapath = './results/LinkCapSweeps/'
    plt.close()
    if interactive:
        plt.ion()
    scalefactors = np.linspace(0,1.5,31)
    ydatalabels = ['BE', 'BC']
    modes = ['lin', 'sqr']
    subplot_counter = 1
    for ydatalabel in ydatalabels:
        for m in modes:
            ydata = []
            for a in scalefactors:
                capstr = str(a) + 'q99'
                fc = FlowCalculation('Europe', 'aHE', capstr, m)
                DC_fc = FlowCalculation('Europe', 'aHE', capstr, 'DC_' + m)
                filename = str(fc)+'.pkl'
                DC_filename = str(DC_fc)+'.pkl'
                ydata.append(sum(get_data(filename, ydatalabel, datapath))\
                            /sum(get_data(DC_filename, ydatalabel, datapath)))
            ax = plt.subplot(2,1,subplot_counter)
            ax.plot(scalefactors, ydata, label=get_pretty_mode(m))
        ax.legend(loc=4)
        ax.set_xlim(0,1.5)
        if ydatalabel=='BE':
            ax.set_ylabel(\
                    r'$E_B^\mathrm{min.\,diss.}/E_B^\mathrm{phase\,angle}$')
        else:
            ax.set_ylabel(\
                    r'$C_B^\mathrm{min.\,diss.}/C_B^\mathrm{phase\,angle}$')
        ax.set_xlabel(r'$\beta$')
        subplot_counter += 1
    plt.tight_layout()
    if not interactive:
        plt.savefig('./results/figures/BEBCsubplotcomparison.pdf')


def plot_bal_vs_trans(ydatalabel='BE', mode='lin', interactive=True,
        figfilename=None, savepath='./results/figures/BalvsTrans/',
        beta_TC=False, omitRolando=False):
    """ Makes a plot of Backup energy (BE) or backup capacity (BC)
        as a function of transmission capacity, for Rolando's and
        the new flow implementation.

        """

    datapath = './results/LinkCapSweeps/'
    plt.close()
    if interactive:
        plt.ion()

    scalefactors = np.linspace(0,1.5,31)
    copperTC = sum(np.load('./results/Europe_copper_linkcaps_'+mode+'.npy'))
    TC = copperTC*scalefactors/1e3
    total_mean_load = sum(np.load('./results/Europe_meanloads.npy'))
    modes = [mode, 'DC_' + mode]
    if omitRolando:
        modes = ['DC_' + mode]

    for m in modes:
        ydata = []
        for a in scalefactors:
            capstr = str(a) + 'q99'
            fc = FlowCalculation('Europe', 'aHE', capstr, m)
            filename = str(fc)+'.pkl'
            ydata.append(sum(get_data(filename, ydatalabel, datapath))\
                            /total_mean_load)
        if not beta_TC:
            plt.plot(TC, ydata,\
                         label=fc.pretty_solvermode()) # TC in GW
        else:
            plt.plot(scalefactors, ydata,\
                         label=fc.pretty_solvermode())
    plt.plot(TC, np.min(ydata)*np.ones_like(TC), '--')
    plt.ylim(0,1.1*max(ydata))
    if not omitRolando:
        plt.legend(loc=4)
    if not beta_TC:
        plt.xlabel('Total transmission capacity [GW]')
        plt.xlim(0,max(TC))
    else:
        plt.xlabel(r'$\beta$')
        plt.xlim(0,1.5)

    if ydatalabel=='BE':
        plt.ylabel('Backup energy [normalized]')
    elif ydatalabel=='BC':
        plt.ylabel('Backup capacity [normalized]')

    if not figfilename:
        figfilename = ydatalabel + 'vsTC_' + mode + '.pdf'
    if not interactive:
        plt.savefig(savepath+figfilename)
        plt.close()

def plot_RF_vs_DCF(capacities, mode, link_number, interactive=True,\
        path='./results/LinkCapSweeps/',\
        savepath='./results/figures/RFvsDCF_plots/', figfilename=None):
    """ Simply plots Rolando's flow as a function of the new DC flow
        for the whole time series. Alpha-values is chosen so that 20 points
        on top of each other is fully colored.

        """

    plt.close()
    if interactive:
        plt.ion()

    DC_F = np.load(''.join([path, 'Europe_aHE_', capacities, '_DC_',\
                           mode, '_flows.npy']))[link_number]

    R_F = np.load(''.join([path, 'Europe_aHE_', capacities, '_',\
                           mode, '_flows.npy']))[link_number]

    plt.plot(DC_F, R_F, '.', alpha=0.05)
    plt.plot(np.linspace(min(DC_F), max(DC_F), 10),\
            np.linspace(min(DC_F), max(DC_F), 10), 'red')

    plt.xlabel(r'$F^{\delta}_l$' + ' [MW]')
    plt.ylabel(r'$F^{F^2}_l$' + ' [MW]')
    plt.title(all_links[link_number] + ': ' + mode + ' ' + capacities)

    if not figfilename:
        figfilename = ''.join([all_links[link_number].replace(' ',''),\
                               '_', mode, '_',  capacities, '.png'])
    if not interactive:
        plt.savefig(savepath+figfilename)
        plt.close()

    return DC_F, R_F


def explain_peak_scatter_plot(filename='Europe_aHE_0.1q99_DC_sqr_flows.npy',\
                              peak_value=394.476, links=range(6),\
                              link_with_peak=0):
    """ This function plots the flows on the specified links in which
        the flow on a certain link (link_with_peak) has a certain value
        (peak_value).
        This allows to check whether peaks are a result of links maxing
        out and locking each other in.

        """

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


def plot_flow_corr_mesh(mode='sqr', interactive=True, sort_by_linkcaps=True,
                savepath='./results/figures/', figfilename=None):
    """ This function plots the linear correlation coefficient between
        Rolando's flow and the new DC flow, for all links, for
        constraint scale factors beta between 0.05 and 1.5.

        """

    plt.close()
    if interactive:
        plt.ion()

    linkcaps = np.load('./results/Europe_copper_linkcaps_'+mode+'.npy')

    beta = np.linspace(0.05, 1.5, 30) # scale factor for the link capacities
    corr_coeff = np.empty((50,len(beta)))
    for i in range(len(beta)):
        DC_filename = 'Europe_aHE_'+str(beta[i])+'q99_DC_'+mode+'_flows.npy'
        R_filename = 'Europe_aHE_'+str(beta[i])+'q99_'+mode+'_flows.npy'
        DC_F = np.load('./results/LinkCapSweeps/'+DC_filename)
        R_F = np.load('./results/LinkCapSweeps/'+R_filename)

        for link in range(50):
            corr_coeff[link, i] = np.corrcoef(DC_F[link], R_F[link])[0,1]

    if sort_by_linkcaps:
        plt.pcolormesh(corr_coeff[linkcaps.argsort(),:])
        plt.yticks(0.5+np.arange(50), \
                    np.array(all_links_ISO2)[linkcaps.argsort()], fontsize=6)
    else:
        plt.pcolormesh(corr_coeff)
        plt.yticks(0.5+np.arange(50), all_links_ISO2, fontsize=6)

    plt.colorbar()
    plt.xticks(20*np.linspace(0,1.5,7),\
                [str(beta) for beta in np.linspace(0,1.5,7)])
    plt.xlabel(r'$\beta$')

    if mode=='sqr':
        plt.title('Pearson correlation, synchronized flow')
    elif mode=='lin':
        plt.title('Pearson correlation, localized flow')

    if not figfilename:
        figfilename = ''.join(['Flowcorr_', mode, '_sorted',\
                               str(sort_by_linkcaps), '.pdf'])
    if not interactive:
        plt.savefig(savepath+figfilename)

    return corr_coeff


def link_caps_barplot(mode='lin'):
    """ This function creates a barchart of the link capacities, found
        as 99% quantiles of the unconstrained flow, given a mode: lin or sqr.

        """

    plt.close()
    linkcaps = np.load('./results/Europe_copper_linkcaps_'+mode+'.npy')
    left = np.arange(50)
    plt.bar(left,np.sort(linkcaps))
    plt.xticks(left+0.5, np.array(all_links)[linkcaps.argsort(),:],\
            rotation=90, size=6)
    plt.plot(np.arange(51), np.ones(51)*np.mean(linkcaps), 'red')

    plt.title('99% quantile link capacities: ' + mode)
    plt.savefig('./results/figures/linkcaps_' + mode + '.pdf')


def conditioned_flow_diff_dist(capacities, mode, link_number,\
        condition_interval, path='results/LinkCapSweeps/', interactive=True,
        bins=200, normed=False, savepath='results/figures/FlowDiffHists/',
        figfilename=None):
    """ This function makes a plot of the distribution of the difference
        between Rolando's flow and the new DC flow, conditioned on the
        DC flow being in a certain interval. This interval is scaled to
        the limits of the flow, so the whole range of flows is (-1.0,1.0).

         Example
        -------
        conditioned_flow_difference_dist('0.1q99', 'lin', 1, (-1,-0.8))

        """
    plt.close()
    if interactive:
        plt.ion()

    DC_F = np.load(''.join([path, 'Europe_aHE_', capacities, '_DC_',\
                           mode, '_flows.npy']))[link_number]

    R_F = np.load(''.join([path, 'Europe_aHE_', capacities, '_',\
                           mode, '_flows.npy']))[link_number]

    diff = R_F - DC_F
    abs_scale = np.max(DC_F)
    lower = condition_interval[0]*abs_scale # left side of bin
    upper = condition_interval[1]*abs_scale # right side of bin

### only take the right end of the interval if it the last bin
    if condition_interval[1]!=1.0:
        conditioned_diff = diff[\
                np.where(np.logical_and(DC_F>=lower, DC_F<upper))]
    else:
        conditioned_diff = diff[\
                np.where(np.logical_and(DC_F>=lower, DC_F<=upper))]

    value, count = myhist(conditioned_diff, bins=bins, normed=normed)
    plt.plot(value, count)
    plt.xlabel(r'$(F_l^{F^2} - F_l^\delta)|F_l^\delta$')
    plt.ylabel('Count')
    plt.title(''.join([all_links[link_number], ': ', mode, ' ',
                       capacities, ', ',
                       str(condition_interval[0]),
                       r'$\leq F_l^\delta/F_l^\mathrm{max} <$',
                       str(condition_interval[1])]))

    if not figfilename:
        figfilename = ''.join([all_links[link_number].replace(' ', ''), '_',\
                       mode, '_', capacities, '_', str(condition_interval[0]),\
                       'to', str(condition_interval[1]), '.png'])
    if not interactive:
        plt.savefig(savepath+figfilename)


def make_selected_flowdiffhists(condition_bin_number=20,\
                        links = links_of_interest, \
                        caps_scalefactors=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5]):
    """ This function uses the function conditioned_flow_diff_dist(...)
        to make some specific plots of the distribution of the difference
        of the Rolando's flow and the new DC flow.

        """

    modes = ['lin', 'sqr']
    capacities = [str(a) + 'q99' for a in caps_scalefactors]

    bin_edges = np.linspace(-1, 1, condition_bin_number + 1)
    intervals = [(bin_edges[i], bin_edges[i+1]) \
                    for i in range(condition_bin_number)]
    intervals.append((-1.0, 1.0))
    link_numbers = [all_links.index(l) for l in links]

    for l in link_numbers:
        for c in capacities:
            for m in modes:
                for i in intervals:
                    conditioned_flow_diff_dist(capacities=c, mode=m,\
                            link_number=l, condition_interval=i,\
                            interactive=False)


def uncond_diff_moments_mesh(mode='lin', moment_number=1,\
                             sort_by_linkcaps=False, interactive=True,\
                             savepath='./results/figures/', figfilename=None,
                             std=False):
    """ This function plots the 1st or 2nd moment of the distribution of
        the difference between Rolando's flow and the new DC flow.
        All links are shown, for beta (constraint scale factors) from 0-1.5.

        """

    plt.close()
    if interactive:
        plt.ion()

    linkcaps = np.load('./results/Europe_copper_linkcaps_'+mode+'.npy')

    beta = np.linspace(0.05, 1.5, 30) # scale factor for the link capacities
    moments = np.empty((50,len(beta)))
    stds = np.empty((50, len(beta)))
    for i in range(len(beta)):
        DC_filename = 'Europe_aHE_'+str(beta[i])+'q99_DC_'+mode+'_flows.npy'
        R_filename = 'Europe_aHE_'+str(beta[i])+'q99_'+mode+'_flows.npy'
        DC_F = np.load('./results/LinkCapSweeps/'+DC_filename)
        R_F = np.load('./results/LinkCapSweeps/'+R_filename)

        diff = R_F - DC_F
        for link in range(50):
            moments[link, i] = np.mean(np.power(diff[link], moment_number))\
                                /np.power(linkcaps[link], moment_number)
            stds[link, i] = np.std(diff[link])/linkcaps[link]

    if sort_by_linkcaps:
        if not std:
            plt.pcolormesh(moments[linkcaps.argsort(),:])
        else:
            plt.pcolormesh(stds[linkcaps.argsort(),:])
        plt.yticks(0.5+np.arange(50), \
                    np.array(all_links_ISO2)[linkcaps.argsort()], fontsize=6)
    else:
        if not std:
            plt.pcolormesh(moments)
        else:
            plt.pcolormesh(stds)
        plt.yticks(0.5+np.arange(50), all_links_ISO2, fontsize=6)

    plt.colorbar()
    plt.xticks(20*np.linspace(0,1.5,7),\
            [str(beta) for beta in np.linspace(0,1.5,7)])
    plt.xlabel(r'$\beta$')

    if moment_number==1:
        plt.title(get_pretty_mode(mode) + ': '\
                 + r'$\langle F_l^{F^2} - F_l^\delta\rangle$'\
                 + ' [normalized]')
    if moment_number==2:
        plt.title(get_pretty_mode(mode) + ': '\
                 + r'$\langle (F_l^{F^2} - F_l^\delta)^2\rangle$'\
                 + ' [normalized]')
    if std:
        plt.title(get_pretty_mode(mode) + ':\n '\
         + 'std' + \
         r'$\left( F_l^\mathrm{min.\,diss.}-F_l^\mathrm{phase\,angle} \right)$'\
                 + ' [normalized]')


    if not figfilename:
        figfilename = ''.join(['FlowDiffMoment_', str(moment_number), '_',\
                               mode, '_sorted', str(sort_by_linkcaps), '.pdf'])
        if std:
            figfilename = ''.join(['FlowDiffstd_',\
                               mode, '_sorted', str(sort_by_linkcaps), '.pdf'])
    if not interactive:
        plt.savefig(savepath+figfilename)


def mean_abs_flow_vs_link_cap(solvermode, link_number):
    plt.close('all')
    plt.ion()

    linkcaps = np.load('./results/Europe_copper_linkcaps_'\
                         + solvermode[-3:] + '.npy')

    beta = np.linspace(0.05, 1.5, 30) # scale factor for the link capacities

    mean_abs_flow = []

    for a in beta:
        capstr = str(a) + 'q99'
        F = PCA.load_flows(capstr, solvermode)[link_number]
        Cl = linkcaps[link_number]*a
        mean_abs_flow.append(np.mean(np.abs(F))/Cl)

    plt.plot(beta, mean_abs_flow, '.')

def flow_characteristic_mesh(solvermode='DC_lin', char='mean_abs_flow', interactive=True, sort_by_linkcaps=True,
                savepath='./results/figures/', figfilename=None):
    plt.close()
    if interactive:
        plt.ion()

    linkcaps = np.load('./results/Europe_copper_linkcaps_' \
                        + solvermode[-3:] + '.npy')

    beta = np.linspace(0.05, 1.5, 30) # scale factor for the link capacities
    flow_char = np.empty((50,len(beta)))
    for i in range(len(beta)):
        filename = 'Europe_aHE_'+str(beta[i])+'q99_'+solvermode+'_flows.npy'
        F = np.load('./results/LinkCapSweeps/'+filename)

        if char == 'mean_abs_flow':
            for link in range(50):
                fplus = beta[i]*linkcaps[link]
                flow_char[link, i] = np.mean(np.abs(F[link]))/fplus
        elif char == 'maxed_prob':
            for link in range(50):
                flow_char[link, i] = get_maxed_out_prob(F, link, 2*1e-3)
        else:
            print \
             "Error: kwarg char must either 'mean_abs_flow' or 'maxed_prob'"

    if sort_by_linkcaps:
        plt.pcolormesh(flow_char[linkcaps.argsort(),:])
        plt.yticks(0.5+np.arange(50), \
                    np.array(all_links_ISO2)[linkcaps.argsort()], fontsize=6)
    else:
        plt.pcolormesh(flow_char)
        plt.yticks(0.5+np.arange(50), all_links_ISO2, fontsize=6)

    plt.colorbar()
    plt.xticks(20*np.linspace(0,1.5,7),\
                [str(beta) for beta in np.linspace(0,1.5,7)])
    plt.xlabel(r'$\beta$')


    if char == 'mean_abs_flow':
        plt.title(get_pretty_imp_and_mode(solvermode) + ': ' \
                                    + r'$\langle|F_l|\rangle /f_l^+ $')
    elif char == 'maxed_prob':
        plt.title(get_pretty_imp_and_mode(solvermode) + ': ' \
                                    + r'$p(|F_l| = f_l^+)$')

    if not figfilename:
        figfilename = ''.join([char, '_', solvermode, '_sorted',\
                               str(sort_by_linkcaps), '.pdf'])
    if not interactive:
        plt.savefig(savepath+figfilename)


### Principal component analysis plots #############################
def Farmer_crit_plot(capacities, solvermode, filename=None, interactive=True,
        title=True, curvefit=False):
    plt.close('all')
    plt.ioff()
    if interactive:
        plt.ion()
    F = PCA.load_flows(capacities, solvermode, filename)
    Phi, K = PCA.FtoPhi(F)
    Nnodes = Phi.shape[0]
    Phi_c, mean_Phi = PCA.center(Phi)
    print "mean_Phi", mean_Phi
    h, Ntilde = PCA.normalize(Phi_c)

    lambdas = np.empty(Nnodes)
    for k in xrange(Nnodes):
        lambdas[k] = PCA.get_principal_component(h, k)[0]
        xi = PCA.get_xi_weight(h, k)
        assert(np.mean(xi**2)-lambdas[k] <= 1e-8)
    print "cumsum lambdas", np.cumsum(lambdas)

    loglambdas = [np.log(lam) for lam in lambdas]


    # curvefit
    if curvefit:
        fitfunc = lambda x, a, b: a*x+b
        ks = 1+np.arange(Nnodes)
        fitparams, fitcov = fit(fitfunc, ks[7:25], loglambdas[7:25])
        print fitparams, fitcov
        plt.plot(np.arange(31),\
                fitfunc(np.arange(31), fitparams[0], fitparams[1]), color=red)


    plt.plot(1+np.arange(Nnodes), loglambdas, '.', color=blue)

    plt.ylim(-12,0)
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\ln \lambda_k$')
    if title:
        plt.title(solvermode + ': ' + capacities)
    figfilename = 'lambdavsk_' + solvermode + '_' + capacities + '.pdf'
    if not interactive:
        plt.savefig('./results/figures/FarmerCritPlots/' + figfilename)
    return lambdas


def make_PC_giffigs(comp_number, solvermode):
    capacities = [str(a) + 'q99' for a in np.linspace(0.05, 1.5, 30)]
    for c in capacities:
        F = PCA.load_flows(c, solvermode)
        Phi, K = PCA.FtoPhi(F)
        Nnodes = Phi.shape[0]
        Phi_c, mean_Phi = PCA.center(Phi)
        h, Ntilde = PCA.normalize(Phi_c)
        lambd, PC = PCA.get_principal_component(h, comp_number)

        Phi_PC = PCA.unnormalize_uncenter(PC, Ntilde, mean_Phi)
        F_PC = PCA.PhitoF(Phi_PC, K)

        normed_Phi_PC = Phi_PC/np.sqrt(np.sum(Phi_PC**2))

        plt.figure()
        left = np.arange(len(Phi_PC))
        plt.bar(left, normed_Phi_PC)
        plt.xticks(left + 0.5, all_countries_ISO2, rotation=90)
        plt.ylabel(r'$\Phi_n$' + ' [MW]')
        plt.ylim((-1, 1))
        plt.title(solvermode + ': ' + c + ', k=' + str(comp_number+1))
        no = str(capacities.index(c))
        figfilename = solvermode + '_k' + str(comp_number+1) + '_' + no + '.png'
        plt.savefig('./results/figures/PCbarplots/' + figfilename)
        plt.close()


def initial_explore_PC_F(capacities, solvermode, comp_number, filename=None):
    plt.close('all')
    plt.ion()
    F = PCA.load_flows(capacities, solvermode, filename)
    Phi, K = PCA.FtoPhi(F)
    Nnodes = Phi.shape[0]
    Phi_c, mean_Phi = PCA.center(Phi)
    h, Ntilde = PCA.normalize(Phi_c)
    lambd, PC = PCA.get_principal_component(h, comp_number)

    Phi_PC = PCA.unnormalize_uncenter(PC, Ntilde, mean_Phi)
    F_PC = PCA.PhitoF(Phi_PC, K)

    left1 = np.arange(len(F_PC))
    plt.bar(left1, F_PC)
    plt.xticks(left1 + 0.5, all_links, rotation=90, size=6)
    plt.ylabel(r'$F_l$' + ' [MW]')
    plt.title(solvermode + ': ' + capacities + ', k=' + str(comp_number+1))

    plt.figure()
    left2 = np.arange(len(Phi_PC))
    plt.bar(left2, Phi_PC)
    plt.xticks(left2 + 0.5, all_countries, rotation=90)
    plt.ylabel(r'$\Phi_n$' + ' [MW]')
    plt.title(solvermode + ': ' + capacities + ', k=' + str(comp_number+1))

    xi = PCA.get_xi_weight(h, comp_number)
    plt.figure()
    plt.hist(xi, bins=530, ec='none', normed=True)
    plt.xlabel(r'$\xi_' + str(comp_number+1) + '$')
    plt.ylabel(r'$p(\xi_' + str(comp_number+1) + ')$')
    plt.title(solvermode + ': ' + capacities + ', k=' + str(comp_number+1))


def xi_hist(capacities, solvermode, comp_number, filename=None,
                figfilename=None, savepath='./results/figures/xihists/',
                interactive=True):
    plt.close('all')
    if interactive:
        plt.ion()

    F = PCA.load_flows(capacities, solvermode, filename)
    Phi, K = PCA.FtoPhi(F)
    Nnodes = Phi.shape[0]
    Phi_c, mean_Phi = PCA.center(Phi)
    h, Ntilde = PCA.normalize(Phi_c)
    xi = PCA.get_xi_weight(h, comp_number)
    plt.figure()
    plt.hist(xi, bins=530, ec='none', normed=True)
    plt.xlabel(r'$\xi_' + str(comp_number+1) + '$')
    plt.ylabel(r'$p(\xi_' + str(comp_number+1) + ')$')
    plt.title(solvermode + ': ' + capacities + ', k=' + str(comp_number+1))

    if not figfilename:
        figfilename = 'xiHist_'+solvermode+'_'+capacities+'_k'\
                       +str(comp_number+1)+'.pdf'
    if not interactive:
        plt.savefig(savepath+figfilename)


def lin_vs_sqr_allcomp_barplot(interactive=False):
    plt.close('all')
    if interactive:
        plt.ion()


    plt.figure(dpi=400, figsize=(10.2, 6))


    modes = ['DC_lin', 'DC_sqr']
    #prepare colors
    colors = []
    country_colordict = {'DE':black, 'FR':'#0055A4', 'IT':'#009246',\
            'GB':'#003399', 'ES':'#AA1518', 'SE':'#ffc200'}
    for c in all_countries_ISO2:
        if c in country_colordict.keys():
            colors.append(country_colordict[c])
        else:
            colors.append(darkgrey)


    lin_PCs = []

    for mode in modes:
        # about the PCA
        F = PCA.load_flows(capacities='copper', solvermode=mode)
        Phi, K = PCA.FtoPhi(F)
        Nnodes = Phi.shape[0]
        Phi_c, mean_Phi = PCA.center(Phi)
        h, Ntilde = PCA.normalize(Phi_c)

        # for plotting
        group_left = np.arange(1,8) - 0.5
        group_width = 0.8
        bin_width = group_width/30

        for comp_number in range(7):
            # PCA
            lambd, PC = PCA.get_principal_component(h, comp_number)
            Phi_PC = PCA.unnormalize_uncenter(PC, Ntilde, mean_Phi)
            norm_Phi_PC = Phi_PC/np.sqrt(np.sum(Phi_PC**2))
            if mode=='DC_lin':
                lin_PCs.append(norm_Phi_PC)
                if comp_number==0:
                    PCsortby = norm_Phi_PC
                    sorted_colors = [colors[i] for i in np.argsort(PCsortby)]
            # plotting
            bin_left = group_width*np.arange(30)/(30) + group_left[comp_number]
            ax = plt.subplot(3,1,modes.index(mode)+1)
            ax.bar(bin_left, norm_Phi_PC[np.argsort(PCsortby)],\
                            width=bin_width, linewidth=0, color=sorted_colors)
            ax.set_title(get_pretty_imp_and_mode(mode))
            ax.set_ylim(-1,1)
            ax.set_xlim(0.4,7.4)
            if mode=='DC_sqr':
                ax.set_ylabel(r'$\Phi_n^k$' + ' [normalized]')
            ax.tick_params(axis='x', labelbottom='off')


            ## handling the difference
            if mode=='DC_sqr':
                diff_PC = lin_PCs[comp_number] - norm_Phi_PC
                ax = plt.subplot(3,1,3)

                ax.bar(bin_left, diff_PC[np.argsort(PCsortby)],\
                            width=bin_width, linewidth=0, color=sorted_colors)
                ax.set_ylim(-1,1)
                ax.set_xlim(0.4,7.4)
                ax.set_xlabel(r'$k$')
                colored_countries = [ 'IT', 'ES', 'FR', 'SE', 'GB', 'DE']
                rectangles = [plt.Rectangle((0,0), 1, 1,\
                        fc=country_colordict[c]) for c in colored_countries]
                plt.legend(rectangles, colored_countries,\
                                     ncol=len(colored_countries), \
                                     prop={'size':13}, loc=4)
                ax.set_title('Difference')

        if not interactive:
            plt.savefig('results/figures/PCA_lin_vs_sqr_allcomp.pdf')




def PC_graph(capacities, solvermode, comp_number, filename=None,\
        figfilename=None, savepath='./results/figures/PCgraphs/', notitle=True):
    plt.close('all')
    plt.ion()

    F = PCA.load_flows(capacities=capacities, solvermode=solvermode,\
                        filename=filename)
    Phi, K = PCA.FtoPhi(F)
    Nnodes = Phi.shape[0]
    Phi_c, mean_Phi = PCA.center(Phi)
    h, Ntilde = PCA.normalize(Phi_c)
    lambd, PC = PCA.get_principal_component(h, comp_number)

    Phi_PC = PCA.unnormalize_uncenter(PC, Ntilde, mean_Phi)
    print Phi_PC
    F_PC = PCA.PhitoF(Phi_PC, K)

    if not notitle:
        title = solvermode + ': ' + capacities + ' k=' + str(comp_number+1)
    else:
        title=None


    if not figfilename:
        figfilename = 'PCgraph_'+solvermode+'_'+capacities+'_k'\
                       +str(comp_number+1)+'.pdf'
    make_europe_graph(lambd, comp_number,\
            link_weights=F_PC, node_weights=Phi_PC, title=title, \
            figfilename=figfilename, savepath=savepath)



def make_europe_graph(lambd, comp_number, link_weights, node_weights,\
        figfilename, savepath, title=None):
    plt.ioff()
    print node_weights
    print len(node_weights)
    G = nx.Graph()
    nodelist = all_countries_ISO2
    linklist = [[link[0:2], link[-2:], all_links_ISO2.index(link)] \
                for link in all_links_ISO2]

    print linklist
    for n in nodelist:
        G.add_node(n)

    for l in linklist:
        G.add_edge(l[0], l[1], weight = link_weights[l[2]])

    pos={}

    pos['AT']=[0.55,0.45]
    pos['FI']=[.95,1.1]
    pos['NL']=[0.40,0.85]
    pos['BA']=[0.65,0.15]
    pos['FR']=[0.15,0.60]
    pos['NO']=[0.5,1.1]
    pos['BE']=[0.275,0.775]
    pos['GB']=[0.10,1.05]
    pos['PL']=[0.75,0.8]
    pos['BG']=[0.9,0.0]
    pos['GR']=[0.7,0.0]
    pos['PT']=[0.0,0.15]
    pos['CH']=[0.4,0.45]
    pos['HR']=[0.75,0.3]
    pos['RO']=[1.0,0.15]
    pos['CZ']=[0.75,0.60]
    pos['HU']=[1.0,0.45]
    pos['RS']=[0.85,0.15]
    pos['DE']=[0.45,0.7]
    pos['IE']=[0.0,0.95]
    pos['SE']=[0.75,1.0]
    pos['DK']=[0.5,0.875]
    pos['IT']=[0.4,0.2]
    pos['SI']=[0.55,0.3]
    pos['ES']=[0.15,0.35]
    pos['LU']=[0.325,0.575]
    pos['SK']=[0.90,0.55]
    pos['EE']=[1.03,0.985]
    pos['LV']=[0.99,0.85]
    pos['LT']=[0.925,0.72]


    redgreendict = {'red':[(0.0, 1.0, 1.0), (0.5, 1.0, 1.0) ,(1.0, 0.0, 0.0)],
                    'green':[(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
                    'blue':[(0.0, 0.2, 0.0), (0.5, 1.0, 1.0), (1.0, 0.2, 0.0)]}


    cmap = LinearSegmentedColormap('redgreen', redgreendict, 1000)

    fig = plt.figure(dpi=400, figsize=(0.85*dcolwidth,0.85*dcolwidth*0.8))
    ax1 = fig.add_axes([0.05, 0.08, 0.9, 0.08])
    ax2 = fig.add_axes([-0.05, 0.15, 1.1, 0.95])

    #span = 2*np.max(np.abs(node_weights))
    #norm_node_weights = [w/span+0.5 for w in node_weights]
    # now the scale shows the actual phi normalized to unit length
    # the offset and factor of 2 is because the colormap is defined
    # from 0 to 1
    phi_length = np.sqrt(np.sum(node_weights**2))
    norm_node_weights = [w/(2*phi_length) + 0.5 for w in node_weights]

    print norm_node_weights
    node_colors = [cmap(w) for w in norm_node_weights]
    nx.draw_networkx_nodes(G, pos, node_size=400, nodelist=nodelist,\
            node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_size=10.8, font_color='k', \
                            font_family='sans-serif')

    maxflow = np.max(np.abs(link_weights))
    no_arrow_limit = 0.05*maxflow
    arrow_length = 0.05
    for l in linklist:
        if -no_arrow_limit<link_weights[l[2]]<no_arrow_limit: continue
        if link_weights[l[2]]>=no_arrow_limit:
            x0 = pos[l[0]][0]
            y0 = pos[l[0]][1]
            x1 = pos[l[1]][0]
            y1 = pos[l[1]][1]
        if link_weights[l[2]]<=-no_arrow_limit:
            x1 = pos[l[0]][0]
            y1 = pos[l[0]][1]
            x0 = pos[l[1]][0]
            y0 = pos[l[1]][1]
        dist = np.sqrt((x0-x1)**2 + (y0-y1)**2)
        plt.arrow(x0, y0, 0.4*(x1-x0), 0.4*(y1-y0), fc=blue, ec=blue,\
                    head_width= 1.2*arrow_length*(abs(link_weights[l[2]]))/maxflow,\
                    head_length = arrow_length)


    nx.draw_networkx_edges(G, pos, edgelist = G.edges(), edge_color=blue)

    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap, orientation='vhorizontal')
    cb1.set_ticks([0, 0.5, 1])
    cb1.set_ticklabels(['-1', '0', '1'])
    strk = str(comp_number+1)
    ax1.set_xlabel(r'$\Phi_n^{(' + strk + ')}$' + ' [normalized]')
    ax1.xaxis.set_label_position('top')
    ax1.set_xticks('none')
    ax2.axis('off')
    strpercentlambd = '%2.1f' % (100*lambd)
    lambdtxt = r'$\lambda_'+strk+'='+strpercentlambd+'\%$'
    props = dict(boxstyle='round', facecolor='w')
    ax2.text(-0.1, 1.15, lambdtxt, bbox=props)
    if title!=None:
        fig.suptitle(title)
    fig.savefig(savepath+figfilename)

    return G

def compare_linsqr_PCmaps():
    plt.close('all')
    fig = plt.figure()

    modes = ['DC_lin', 'DC_sqr']
    comp_numbers = range(7)

    for mode in modes:
        F = PCA.load_flows(capacities='copper', solvermode=mode)
        Phi, K = PCA.FtoPhi(F)
        Nnodes = Phi.shape[0]
        Phi_c, mean_Phi = PCA.center(Phi)
        h, Ntilde = PCA.normalize(Phi_c)
        for comp_number in comp_numbers:
            #PCA
            lambd, PC = PCA.get_principal_component(h, comp_number)
            Phi_PC = PCA.unnormalize_uncenter(PC, Ntilde, mean_Phi)

            #plotting
            subplotindex = 1 + comp_number + 7*modes.index(mode)
            print comp_number, mode
            print subplotindex
            ax = plt.subplot(2, 7, subplotindex)
            plot_europe_map(Phi_PC, ax)
            strk = str(comp_number+1)
            strpercentlambd = '%2.1f' % (100*lambd)
            ax.set_xlabel(r'$\lambda_' + strk + '=' + strpercentlambd + '\%$')
            if np.mod(subplotindex, 7)==4:
                mode_labels = ['Localized flow', 'Synchronized flow']
                label_index = subplotindex/7 # integer division
                ax.set_title(mode_labels[label_index])
            plt.box(on='off')

    # colorbar
    fig.subplots_adjust(bottom=0.49)
    cbar_ax = fig.add_axes([0.25, 0.37, 0.5, 0.05])

    redgreendict = {'red':[(0.0, 1.0, 1.0), (0.5, 1.0, 1.0) ,(1.0, 0.0, 0.0)],
                    'green':[(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
                    'blue':[(0.0, 0.2, 0.0), (0.5, 1.0, 1.0), (1.0, 0.2, 0.0)]}
    cmap = LinearSegmentedColormap('redgreen', redgreendict, 1000)
    cb1 = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap,\
                                             orientation='vhorizontal')
    cb1.set_ticks([0, 0.5, 1])
    cb1.set_ticklabels(['-1', '0', '1'])
    cbar_ax.set_xlabel(r'$\Phi_n^{(k)}$' + ' [normalized]')
    cbar_ax.xaxis.set_label_position('top')
    cbar_ax.set_xticks('none')

    fig.savefig('results/figures/compare_linsqr_maps.pdf')


def plot_europe_map(country_weights, ax=None):

    #plt.close('all')
    #plt.ion()
    #myfig = plt.figure()
    if ax==None:
        ax = plt.subplot(111)
    m = Basemap(llcrnrlon=-10., llcrnrlat=30., urcrnrlon=50., urcrnrlat=72.,\
                        projection='lcc', lat_1=40., lat_2=60., lon_0=20.,\
                                    resolution ='l', area_thresh=1000.,\
                                    rsphere=(6378137.00, 6356752.3142))
    #m.drawcountries(linewidth=0.5)
    m.drawcoastlines(linewidth=0)
    r = shapefile.Reader(\
            r'data/ne_10m_admin_0_countries/ne_10m_admin_0_countries')
    all_shapes = r.shapes()
    all_records = r.records()
    shapes = []
    records = []
    for country in all_countries:
        shapes.append(all_shapes[shapefile_index[country]])
        records.append(all_records[shapefile_index[country]])



    redgreendict = {'red':[(0.0, 1.0, 1.0), (0.5, 1.0, 1.0) ,(1.0, 0.0, 0.0)],
                    'green':[(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
                    'blue':[(0.0, 0.2, 0.0), (0.5, 1.0, 1.0), (1.0, 0.2, 0.0)]}

    cmap = LinearSegmentedColormap('redgreen', redgreendict, 1000)


    weights_length = np.sqrt(np.sum(country_weights**2))
    norm_center_countryweights = [w/(2*weights_length) + 0.5 for w in country_weights]

    country_count = 0
    for record, shape in zip(records, shapes):
        lons, lats = zip(*shape.points)
        data = np.array(m(lons, lats)).T

        if len(shape.parts) == 1:
            segs = [data,]
        else:
            segs = []
            for i in range(1, len(shape.parts)):
                index = shape.parts[i-1]
                index2 = shape.parts[i]
                segs.append(data[index:index2])
            segs.append(data[index2:])

        lines = LineCollection(segs, antialiaseds=(1,))
        lines.set_facecolor(cmap(norm_center_countryweights[country_count]))
        lines.set_edgecolors('k')
        lines.set_linewidth(0.3)
        ax.add_collection(lines)

        country_count += 1


