import numpy as np
import cPickle as pickle
import aurespf.solvers as au
from FlowCalculation import FlowCalculation


class FCResult:
    """ This is a class for storing important data from a
        flow calculation, without saving the entire
        timeseries. This is necessary when doing many
        flow calculation, because saved nodes-object
        (.npz) take up around 150 Mb.
        The class stores the following data for each node
        in the network:
        alpha, gamma, the balancing energy normalized
        to the mean load of the region, balancing capacity
        normalized to the mean load and the transmission
        capacity in GW (if an FlowCalculation object is
        passed to add_instance, and the appropriate copper
        flow file (.npy) is available in the results folder.


        Examples
        --------
        Store the data from the solved nodes object N,
        with flows F in a FCResult file:

        >>> myresults = FCResult("myfile.pkl")
        # if a flowcalculation object, FC, exits for
        # the calculation, passing this allows for the
        # transmission capacity to be stored also.
        >>> myresults.add_instance(N, F, FC)
        >>> myresults.save_results("myfile.pkl")

        """

    def __init__(self, filename, path="./results/"):

        self.cache = []

        try:
            self.__load_results__(filename, path)
        except:
            self.save_results(filename, path)

    def __load_results__(self, filename, path="./results/"):

        with open(path+filename) as currentfile:
            self.cache = pickle.load(currentfile)
            assert(self.cache.__class__ == list),\
                    "Attempt to load wrong data format "\
                    + filename + "may be corrupt."
        print "Loading data from file: ", filename

    def __len__(self):
        return len(self.cache)

    def __str__(self):
        return filename

    def save_results(self, filename, path="./results/"):
        with open(path+filename, 'w') as currentfile:
            pickle.dump(self.cache, currentfile)
        if len(self.cache) == 0:
            print "File: " + filename + " created."
        if len(self.cache) > 0:
            print "Results saved to file: ", filename

    def add_instance(self, N, F, FC):

        inst={}
        inst['alphas'] = [n.alpha for n in N]
        inst['gammas'] = [n.gamma for n in N]
        length_of_timeseries = len(N[0].balancing)

                # save data, that requires a Flowcalculation object for specification
        inst['FlowCalculation'] = FC

        if FC.capacities!='zerotrans':
            inst['BE'] = [np.sum(n.balancing)/\
                              length_of_timeseries for n in N]
            inst['BC'] = [au.get_q(n.balancing, 0.99) for n in N]
        # Save flow histograms of all the flow along the links
            flow_histograms = []
            for link in xrange(F.shape[0]):
                flow, count = myhist(F[link], bins=200)
                flow_histograms.append(np.array([flow, count]))
            inst['flowhists'] = flow_histograms



            if FC.hourly_flowhist:
                # flow histograms, of flow along the links, conditioned
                # on the hour.
                hourly_flow_histograms = []
                for link in xrange(F.shape[0]):
                    hour_hist_list = []
                    for hour in range(24):
                        hour_indices = [i for i in xrange(F.shape[1]) \
                                        if np.mod(i, 24)==hour]
                        hour_flows = F[link][hour_indices]
                        flow, count = myhist(hour_flows, bins=200)
                        hour_hist_list.append(np.array([flow, count]))
                    hourly_flow_histograms.append(hour_hist_list)
                inst['hourly_flowhists'] = hourly_flow_histograms


            if FC.capacities[-3:len(FC.capacities)] == "q99":
                inst['Total_TC'] = TC_from_FC(FC)[0]
                inst['TC'] = TC_from_FC(FC)[1]
            elif FC.capacities == 'copper':
                h0 = au.get_quant_caps(F=F)
                inst['TC'] = h0
                inst['Total_TC'] = np.sum(au.biggestpair(h0))
        else: # that is if this is a zero transmission case
            inst['Total_TC'] = 0
            balancing_timeseries = []
            for n in N:
                balancing_timeseries.append(-au.get_negative(n.mismatch))
            inst['BE'] = [np.sum(bal)/length_of_timeseries\
                          for bal in balancing_timeseries]
            inst['BC'] = [au.get_q(bal, 0.99) for bal in balancing_timeseries]

        self.cache.append(inst)

def TC_from_FC(FC):
    copperFC = FC.copy()
    copperFC.capacities = "copper"
    copperflowfile = ''.join(["results/", str(copperFC),\
                         '_flows.npy'])
    h0 = au.get_quant_caps(filename=copperflowfile)
    # expecting capacities of format: 0.35q99
    scalefactor = float(FC.capacities[0:-3])

    total_TC = scalefactor*np.sum(au.biggestpair(h0))
    return total_TC, h0


def myhist(*args, **kwargs):
    """ Takes any of the same arguments as np.histogram() and returns
        an array of xvalues, corresponding to the centers of the bins
        of the histogram, along with the height of the bins (count).

        """

    count, bins = np.histogram(*args, **kwargs)
    delta = float(bins[1]-bins[0])/2
    xs = bins[:-1] + delta
    return xs, count
