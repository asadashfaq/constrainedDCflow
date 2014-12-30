import numpy as np
import matplotlib.pyplot as plt
import shapefile
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

import aurespf.solvers as au
import PCA_tools as PCA
import figutils as fig
from europe_plusgrid import europe_plus_Nodes


def mismatch_Farmer_crit_plot(\
                    mismatch_filename='Europe_gHO1.0_aHO0.8_mismatches.npy',\
                    interactive=True):

    plt.close('all')
    plt.ioff()
    if interactive:
        plt.ion()

    mismatch = np.load('results/' + mismatch_filename) # a 30 by 280512 matrix
    Nnodes = mismatch.shape[0]
    mismatch_c, mean_mismatch = PCA.center(mismatch)
    h, Ntilde = PCA.normalize(mismatch_c)

    lambdas = np.empty(Nnodes)
    for comp_number in xrange(Nnodes):
        lambdas[comp_number] = PCA.get_principal_component(h, comp_number)[0]
        xi = PCA.get_xi_weight(h, comp_number)
        assert(np.mean(xi**2-lambdas[comp_number] <= 1e-8))


    print lambdas
    plt.semilogy(1+np.arange(Nnodes), lambdas, '.')


def plot_map_mismatch_PC(comp_number, mismatch_filename='Europe_gHO1.0_aHO0.8_mismatches.npy',\
                    interactive=True):
    plt.close('all')
    plt.ioff()
    if interactive:
        plt.ion()

    mismatch = np.load('results/' + mismatch_filename) # a 30 by 280512 matrix
    Nnodes = mismatch.shape[0]
    mismatch_c, mean_mismatch = PCA.center(mismatch)
    h, Ntilde = PCA.normalize(mismatch_c)
    lambd, PC = PCA.get_principal_component(h, comp_number)
    print PC
    mismatch_PC = PCA.unnormalize_uncenter(PC, Ntilde, mean_mismatch)
    print mismatch_PC
    plot_europe_map(mismatch_PC)


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
    for country in fig.all_countries:
        shapes.append(all_shapes[fig.shapefile_index[country]])
        records.append(all_records[fig.shapefile_index[country]])



    redgreendict = {'red':[(0.0, 1.0, 1.0), (0.5, 1.0, 1.0) ,(1.0, 0.0, 0.0)],
                    'green':[(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
                    'blue':[(0.0, 0.2, 0.0), (0.5, 1.0, 1.0), (1.0, 0.2, 0.0)]}

    cmap = fig.LinearSegmentedColormap('redgreen', redgreendict, 1000)


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


def create_approx_mismatch(K,\
        mismatch_filename='Europe_gHO1.0_aHO0.8_mismatches.npy'):

    """ K is the number of principal components that go into the
        approximation. K = 1 means that only the first component
        is used, K=7 that the first 7 are used.

        """

    mismatch = np.load('results/' + mismatch_filename) # a 30 by 280512 matrix
    print mismatch.shape
    Nnodes = mismatch.shape[0]
    mismatch_c, mean_mismatch = PCA.center(mismatch)
    h, Ntilde = PCA.normalize(mismatch_c)

    approx_mismatch = np.zeros_like(mismatch)
    for comp_number in xrange(K):
        lambd, princ_comp = PCA.get_principal_component(h, comp_number)
        print lambd
        mismatch_PC = PCA.unnormalize_uncenter(princ_comp, \
                                                Ntilde, mean_mismatch)
        xi = PCA.get_xi_weight(h, comp_number)

        approx_mismatch += np.outer(mismatch_PC, xi)


    np.save('results/Europe_gHO1.0_aHO0.8_approx_mismatch_K'+str(K)+'.npy',\
             approx_mismatch)
    return


def get_total_backup_energy(N, normalized=True):
    length_of_timeseries = len(N[0].balancing)
    total_mean_load = np.sum([n.mean for n in N])

    absolute_BE = np.sum([np.sum(n.balancing)/length_of_timeseries for n in N])
    normed_BE = absolute_BE/total_mean_load

    return normed_BE


def get_gen_backup_capacity(N, quantile, normalized=True):

    total_mean_load = np.sum([n.mean for n in N])
    total_BC = np.sum([au.get_q(n.balancing, quantile) for n in N])
    normed_BC = total_BC/total_mean_load

    return normed_BC


def get_total_transmission_capacity(F, N, quantile, normalized=True):
    """ Note that this uses a slightly different definition of TC.
        The quantile of the absolute flow is used.

        """

    total_mean_load = np.sum([n.mean for n in N])
    total_TC = np.sum([au.get_q(np.abs(F[i]), quantile) \
                       for i in xrange(F.shape[0])])

    normed_TC = total_TC/total_mean_load

    return normed_TC


def make_table(table_filename):
    base_filename = 'Europe_aHO0.8_copper_lin'
    filename_Ks = [''] + ['_K' + str(K) for K in range(1,8)]
    node_filenames = [base_filename + K + '.npz' for K in filename_Ks]
    flow_filenames = [base_filename + K + '_flows.npy' for K in filename_Ks]

    textfile = open(table_filename, 'w')
    table_data_lines = ['' for i in range(9)]
    quantiles = [0.99, 0.999, 0.9999, 1.0]

    for node_filename in node_filenames:
        N = europe_plus_Nodes(load_filename=node_filename)
        table_data_lines[0] += ' & %2.4f' % get_total_backup_energy(N)

        for i in range(4):
            table_data_lines[i+1] += ' & %2.3f'  %  get_gen_backup_capacity(N,\
                                                      quantile = quantiles[i])

        flow_filename = flow_filenames[node_filenames.index(node_filename)]
        F = np.load('results/'+flow_filename)
        for i in range(4):
            table_data_lines[i+5] += \
                    ' & %2.3f' %  get_total_transmission_capacity(F, N,\
                                                      quantile=quantiles[i])
        del N


    for i in range(9):
        textfile.write(table_data_lines[i])
        textfile.write(r' \\')
        textfile.write('\n')
    textfile.close()


def make_mismatch_PC_mapplot(\
        mismatch_filename='Europe_gHO1.0_aHO0.8_mismatches.npy'):
    plt.close('all')
    plt.ion()
    myfig = plt.figure()

    #prepare PCA
    mismatch = np.load('results/' + mismatch_filename) # a 30 by 280512 matrix
    Nnodes = mismatch.shape[0]
    mismatch_c, mean_mismatch = PCA.center(mismatch)
    h, Ntilde = PCA.normalize(mismatch_c)

    axis = [2] + range(4, 10)
    comp_numbers = np.arange(7)
    props = dict(boxstyle='round', facecolor='w') # texbox props
    for a, comp_number in zip(axis, comp_numbers):
        # PCA
        lambd, PC = PCA.get_principal_component(h, comp_number)
        mismatch_PC = PCA.unnormalize_uncenter(PC, Ntilde, mean_mismatch)
        print 'PC' + str(comp_number + 1)
        print mismatch_PC
        #plotting
        ax = plt.subplot(3,3,a)
        plot_europe_map(mismatch_PC, ax=ax)
        lambd_string = '%2.1f ' % (100.0*lambd)
        label = 'PC' + str(comp_number + 1) + '\n'\
                + r'$\lambda = ' + lambd_string + r'\,\% $'
        ax.text(0.0, 0.9, label, transform=ax.transAxes, fontsize=10,\
                verticalalignment='center', bbox=props)
        plt.box(on='off')

    plt.tight_layout()

    # colorbar
    myfig.subplots_adjust(bottom=0.2)
    cbar_ax = myfig.add_axes([0.05, 0.05, 0.9, 0.1])

    redgreendict = {'red':[(0.0, 1.0, 1.0), (0.5, 1.0, 1.0) ,(1.0, 0.0, 0.0)],
                    'green':[(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
                    'blue':[(0.0, 0.2, 0.0), (0.5, 1.0, 1.0), (1.0, 0.2, 0.0)]}
    cmap = LinearSegmentedColormap('redgreen', redgreendict, 1000)
    cb1 = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap,\
                                             orientation='vhorizontal')
    cb1.set_ticks([0, 0.5, 1])
    cb1.set_ticklabels(['-1', '0', '1'])
    cbar_ax.set_xlabel(r'$\Delta_n^k$' + ' [normalized]')
    cbar_ax.xaxis.set_label_position('top')
    cbar_ax.set_xticks('none')

    plt.savefig('results/figures/PCA_article/' +
                mismatch_filename[:-4] + '.pdf')

    return


def plot_lambdas_vs_alpha():
    plt.close('all')
    plt.ion()
    N = europe_plus_Nodes(alphas=0.8)
    Nnodes = len(N)

    alphas = np.linspace(0, 1, 21)
    lambdas = np.empty((21, 30))
    for alpha in alphas:
        N.set_alphas(alpha*np.ones(Nnodes))
        mismatch = np.empty((Nnodes, len(N[0].mismatch)))
        for i in range(Nnodes):
            mismatch[i] = N[i].mismatch

        mismatch_c, mean_mismatch = PCA.center(mismatch)
        h, Ntilde = PCA.normalize(mismatch_c)
        for comp_number in np.arange(30):
            lambd, PC = PCA.get_principal_component(h, comp_number)
            lambdas[np.where(alphas==alpha)[0][0]][comp_number] = lambd

    del N

    cum_lambds = np.cumsum(lambdas, axis=1)
    for i in reversed(range(Nnodes)):
        colorindex = np.mod(i, len(fig.color_cycle))
        plt.fill_between(alphas, cum_lambds[:,i],\
                         facecolor=fig.color_cycle[colorindex])


    plt.ylim((0.0, 1.0))
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Sum of eigenvalues ' + r'$\lambda_k$')
    # write text bits
    ypos = [0.236, 0.569, 0.730, 0.816, 0.871, 0.941]
    text_bits = [r'$k=' + str(i) + r'$' for i in range(1,6)] + \
                [r'$6 \leq k \leq 30$']
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    for i in range(len(ypos)):
        plt.text(0.88, ypos[i], text_bits[i], fontsize=10, bbox=props)

    plt.savefig('results/figures/PCA_article/cum_lambdas_vs_alpha.pdf')

    return lambdas


