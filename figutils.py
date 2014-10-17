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
        figfilename = all_links[link_number].replace(' ', '') + '_' + mode\
                        + ('_%02i.png'%(caps.index(c)))
        plot_flow_hists(fclist=fclist, link_number=link_number,\
                        interactive=False, figfilename=figfilename,\
                        savepath=savepath,
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
            label = fc.pretty_solvermode()
        if semilogy:
            plt.semilogy(value, count, label=label)
        else:
            plt.plot(value, count, label=label)

    plt.legend()
    if not showbeta:
        plt.title(all_links[link_number])
    else:
        beta_str = get_beta_string(fclist[0].capacities[0:-3])
        plt.title(all_links[link_number] + ': ' + r'$\beta=$' + beta_str)

    plt.xlabel('Power flow [MW]')
    plt.ylabel('Count')

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    if not figfilename:
        figfilename = ''.join([all_links[link_number].replace(' ', ''), '_',\
                                fclist[0].solvermode, '_', \
                                fclist[0].capacities, '.pdf'])
    if not interactive:
        plt.savefig(savepath+figfilename)
        plt.close()


def make_all_bal_vs_trans():
    """ Calls plot_bal_vs_trans(...) in the four relevant cases.

        """
    modes = ['lin', 'sqr']
    ydatalabels = ['BE', 'BC']
    for m in modes:
        for y in ydatalabels:
            plot_bal_vs_trans(ydatalabel=y, mode=m, interactive=False)

    return


def plot_bal_vs_trans(ydatalabel='BE', mode='lin', interactive=True,
        figfilename=None, savepath='./results/figures/BalvsTrans/'):
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
                    np.array(all_links)[linkcaps.argsort()], fontsize=6)
    else:
        plt.pcolormesh(corr_coeff)
        plt.yticks(0.5+np.arange(50), all_links, fontsize=6)

    plt.colorbar()
    plt.xticks(20*np.linspace(0,1.5,7),\
                [str(beta) for beta in np.linspace(0,1.5,7)])
    plt.xlabel(r'$\beta$')

    if mode=='sqr':
        plt.title('Correlation, synchronized flow')
    elif mode=='lin':
        plt.title('Correlation, localized flow')

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
                             savepath='./results/figures/', figfilename=None):
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
    for i in range(len(beta)):
        DC_filename = 'Europe_aHE_'+str(beta[i])+'q99_DC_'+mode+'_flows.npy'
        R_filename = 'Europe_aHE_'+str(beta[i])+'q99_'+mode+'_flows.npy'
        DC_F = np.load('./results/LinkCapSweeps/'+DC_filename)
        R_F = np.load('./results/LinkCapSweeps/'+R_filename)

        diff = R_F - DC_F
        for link in range(50):
            moments[link, i] = np.mean(np.power(diff[link], moment_number))\
                                /np.power(linkcaps[link], moment_number)

    if sort_by_linkcaps:
        plt.pcolormesh(moments[linkcaps.argsort(),:])
        plt.yticks(0.5+np.arange(50), \
                    np.array(all_links)[linkcaps.argsort()], fontsize=6)
    else:
        plt.pcolormesh(moments)
        plt.yticks(0.5+np.arange(50), all_links, fontsize=6)
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

    if not figfilename:
        figfilename = ''.join(['FlowDiffMoment_', str(moment_number), '_',\
                               mode, '_sorted', str(sort_by_linkcaps), '.pdf'])
    if not interactive:
        plt.savefig(savepath+figfilename)


##############################################################################
############## Auxilary functions ############################################
##############################################################################


def get_pretty_mode(mode):
    """ Auxilary function that transforms 'lin' into 'Localized flow'
        and 'sqr' into 'Synchronized flow'.

        """

    return mode.replace('lin', 'Localized').replace('sqr', 'Synchronized')\
                        + ' flow'


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
