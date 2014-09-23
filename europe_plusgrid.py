import numpy as np
import aurespf.solvers as au

def europe_plus_Nodes(admat=None, load_filename=None, full_load=False, alphas=None):


    if not admat and not load_filename:
        admat = './settings/Europeadmat.txt' # only the European countries

    Europe_regions = ['AUT', 'FIN', 'NLD', 'BIH', 'FRA', 'NOR', 'BEL','GBR', \
                 'POL', 'BGR', 'GRC', 'PRT', 'CHE', 'HRV', 'ROU', 'CZE',\
                 'HUN', 'SRB', 'DEU', 'IRL', 'SWE', 'DNK', 'ITA', 'SVN',\
                 'ESP', 'LUX', 'SVK', 'EST', 'LVA', 'LTU']
    all_regions = Europe_regions + ['RU', 'NA', 'ME']

    regions = Europe_regions
    if admat:
        if 'RU' in admat:
            regions.append('RU')
        if 'NA' in admat:
            regions.append('NA')
        if 'ME' in admat:
            regions.append('ME')
    elif load_filename:
        if 'RU' in load_filename:
            regions.append('RU')
        if 'NA' in load_filename:
            regions.append('NA')
        if 'ME' in load_filename:
            regions.append('ME')
    else:
        print("You must provide either an admat or a load_filename")

    files = [''.join([r,'.npz']) for r in regions]

    prefix='VE_'

    if alphas==None:
        all_alphas = np.load("./results/europeplusOptimalmixes.npy")
        alphas = []
        for r in regions:
            alphas.append(all_alphas[all_regions.index(r)])
    if type(alphas)==float or type(alphas)==int:
        homogeneousalpha = alphas
        alphas = homogeneousalpha*np.ones(len(regions))

    assert(len(alphas)==len(regions))

    return au.Nodes(admat=admat, path='./data/', prefix=prefix,
                    files=files, load_filename=load_filename,
                    full_load=full_load, alphas=alphas,
                    gammas=np.ones(len(regions)))
