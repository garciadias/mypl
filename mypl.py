import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import numpy as np
import os
from astropy.coordinates import SkyCoord
from astropy import units as u
import itertools
from sklearn.metrics.cluster import contingency_matrix
from scipy.ndimage.filters import gaussian_filter

def creat_colours():
    from matplotlib.colors import LinearSegmentedColormap
    cmBlues   = LinearSegmentedColormap.from_list('Blues',  ['#dfe6eb','#c0cdd8', '#597a98', '#30597f', '#182c3f'], N=50)
    cmReds    = LinearSegmentedColormap.from_list('Reds' ,  ['#f7e0e0','#f4d3d3', '#df7b7b', '#cb2424', '#791515'], N=50)
    cmGreys   = LinearSegmentedColormap.from_list('Greys',  ['#e7e7e7','#888888', '#555555', '#333333', '#000000'], N=50)
    cmPurples = LinearSegmentedColormap.from_list('Purples',['#d5bcc9','#98597a', '#7f3059', '#652647', '#321323'], N=50)
    cmGreens  = LinearSegmentedColormap.from_list('Greens', ['#d5e5dd','#97bfaa', '#307f56', '#1c4c33', '#0e2619'], N=50)
    cmOranges = LinearSegmentedColormap.from_list('Oranges',['#f4dfc2','#eac086', '#ffad60', '#e59b56', '#895d33'], N=50)
    cmpink_r  = LinearSegmentedColormap.from_list('pink_r', ['#f7d2fa','#f3b4f7', '#e86af0', '#b954c0', '#6f3273'], N=50)
    cmYlGn    = LinearSegmentedColormap.from_list('YlGn',   ['#edf3c1','#d4e266', '#c6d932', '#b8d000', '#6e7c00'], N=50)
    Blues   ='#369af4'#597a98'
    Reds    ='#df7b7b'
    Greys   ='#888888'#555555'
    Purples ='#98597a'#7f3059'
    Greens  ='#51CC8B'#307f56'
    Oranges ='#FD9636'#ffad60'
    pink_r  ='#f3b4f7'#e86af0'
    YlGn    ='#d4e266'#c6d932'
    CMAP = {}
    CMAP['00']= cmBlues
    CMAP['01']= cmReds
    CMAP['02']= cmGreys
    CMAP['03']= cmBlues
    CMAP['04']= cmPurples
    CMAP['05']= cmGreens
    CMAP['06']= cmOranges
    CMAP['07']= cmpink_r
    #CMAP['08']= cmGreens
    CMAP['08']= cmYlGn
    CMAP['09']= cmBlues
    CMAP['10']= cmBlues
    CMAP['11']= cmReds
    CMAP['12']= cmReds
    CMAP['13']= cmGreys
    CMAP['14']= cmReds
    CMAP['15']= cmGreys
    CMAP['16']= cmGreys
    CMAP['17']= cmPurples
    CMAP['18']= cmPurples
    CMAP['19']= cmGreens
    CMAP['20']= cmGreens
    CMAP['21']= cmPurples
    CMAP['22']= cmOranges
    CMAP['23']= cmOranges
    #CMAP['23']= cmGreens
    CMAP['24']= cmReds
    CMAP['25']= cmYlGn
    CMAP['26']= cmOranges
    CMAP['27']= cmPurples
    CMAP['28']= cmBlues
    CMAP['29']= cmGreens
    CMAP['30']= cmYlGn
    CMAP['31']= cmBlues
    COLOR = {}
    COLOR['00']= Blues
    COLOR['01']= Reds
    COLOR['02']= Greys
    COLOR['03']= Blues
    COLOR['04']= Purples
    COLOR['05']= Greens
    COLOR['06']= Oranges
    COLOR['07']= pink_r
    COLOR['08']= 'y'
    COLOR['09']= Blues
    COLOR['10']= Blues
    COLOR['11']= Reds
    COLOR['12']= Reds
    COLOR['13']= Greys
    COLOR['14']= Reds
    COLOR['15']= Greys
    COLOR['16']= Greys
    COLOR['17']= Purples
    COLOR['18']= Purples
    COLOR['19']= Greens
    COLOR['20']= Greens
    COLOR['21']= Purples
    COLOR['22']= Oranges
    COLOR['23']= Oranges
    COLOR['24']= Reds
    COLOR['25']= 'y'
    COLOR['26']= Oranges
    COLOR['27']= Purples
    COLOR['28']= Blues
    COLOR['29']= Greens
    COLOR['30']= 'y'
    COLOR['31']= Blues
    COLOR['32']= Greys
    COLOR['33']= Greys
    COLOR['34']= Greys
    COLOR['35']= Blues
    COLOR['36']= Blues
    COLOR['37']= Blues
    COLOR['38']= Oranges
    COLOR['39']= Oranges
    COLOR['40']= Oranges
    COLOR['41']= Greens
    COLOR['42']= Greens
    COLOR['43']= Greens
    COLOR['44']= Reds
    COLOR['45']= Reds
    COLOR['46']= Reds
    COLOR['47']= pink_r
    COLOR['48']= pink_r
    COLOR['49']= pink_r
    return COLOR, CMAP

def look_on_flags(flag_list, data, classes):
    STRINGS = {}
    NUMBERS = {}
    ORDERS  = {}
    for k in classes:
        Strings = []        
        Numbers = []                                            
        for spec in np.where(data['assign'] == k)[0]:
            tag_list = flag_list[spec].split(',')
            for name in tag_list:
                try:
                    match = Strings.index(name.split(' ')[0])
                    Numbers[match] +=1
                except:
                    Strings.append(name.split(' ')[0])
                    Numbers.append(1)
        Strings = np.array(Strings)
        Numbers = np.array(Numbers)/data['count'][k]
        order = [i[0] for i in sorted(enumerate(Numbers), key=lambda x:x[1])][::-1]
        print('Class %02d'%k)
        for i in order[:10]:
            print('%40s\t%0.2f'%(Strings[i], Numbers[i]*100.))
        STRINGS['C%02d'%k] = Strings 
        NUMBERS['C%02d'%k] = Numbers
        ORDERS['C%02d'%k]  = order
    return STRINGS, NUMBERS, ORDERS

def CH_index(data, masked, transpose=False, do_mask=False, centroid=[]):
    if (transpose): 
        clusters = data['clusters']
        sclusters = data['sclusters']
    else: 
        clusters = data['clusters'].transpose()
        sclusters = data['sclusters'].transpose()
    if (centroid == []): centroid = np.load('/net/tarea/scratch/Rafael/phd/apogee/python/FULL_MEAN.npy')
    if (do_mask): 
        BCSM = (data['count'][:data['nc']]*((centroid[masked] - clusters[:, masked])**2).sum(axis=1)).sum()
        WCSM = (data['count'][:data['nc']]*((sclusters)**2).sum(axis=1)).sum()
    else:                                        
        BCSM = (data['count'][:data['nc']]*((centroid[masked] - clusters)**2).sum(axis=1)).sum()
        WCSM = (data['count'][:data['nc']]*((sclusters)**2).sum(axis=1)).sum()
    norm = (data['count'][:data['nc']].sum() - data['nc'])/(data['nc'] -1)
    CH = (BCSM/(data['nc'] -1))/(WCSM/(data['count'][:data['nc']].sum() - data['nc']))
    return CH, WCSM, BCSM, norm
    

def equat_to_galatic(RA, DEC):
    coo_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
    l = coo_icrs.galactic.l.deg    
    b = coo_icrs.galactic.b.deg    
    return l, b

def projection(ax, RA,Dec,org=0,title='Mollweide projection', projection='aitoff', s=10, marker='o', color='r', fig=None, LABEL='', graph_type='scatter'):
    ''' RA, Dec are arrays of the same length.
    RA takes values in [0,360), Dec in [-90,90],
    which represent angles in degrees.
    org is the origin of the plot, 0 or a multiple of 30 degrees in [0,360).
    title is the title of the figure.
    projection is the kind of projection: 'mollweide', 'aitoff', 'hammer', 'lambert'
    '''
    x = np.remainder(RA+360-org,360) # shift RA values
    ind = x>180
    x[ind] -=360    # scale conversion to [-180, 180]
    x=-x    # reverse the scale: East to the left
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360+org,360)
    #ax = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1, projection=projection, axisbg ='LightCyan')
    if graph_type == 'scatter':
        cs = ax.scatter(np.radians(x),np.radians(Dec), s=s, marker=marker, color=color, alpha=1, label=LABEL)  # convert degrees to radians
    else:
        xmin, xmax, binwidthX = min(x), max(x), (max(x)-min(x))/55.
        ymin, ymax, binwidthY = min(Dec), max(Dec), (max(Dec)-min(Dec))/55.#0.025
        binsX = np.arange(np.radians(xmin), np.radians(xmax), np.radians(binwidthX))
        binsY = np.arange(np.radians(ymin), np.radians(ymax), np.radians(binwidthY))
        H, xedges, yedges = np.histogram2d(np.radians(x), np.radians(Dec), bins=(binsX, binsY))
        extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
        N_CLASS = len(x)
        levels = np.array(find_levels(H, N_CLASS))
        cs = plt.contourf(gaussian_filter(H.transpose(), 0.95), levels, extent=[xedges.min(), xedges.max(), yedges.min(),yedges.max()], cmap=color, alpha=1,extend='max')
    ax.set_xticklabels(tick_labels, fontsize=14)     # we add the scale on the x axis
    tick_labels = np.array([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75])
    #tick_labels = np.remainder(tick_labels+90+org,90)
    ax.set_yticklabels(tick_labels, fontsize=14)     # we add the scale on the x axis
    ax.set_title(title)
    ax.title.set_fontsize(18)
    ax.set_xlabel("l($^o$)")
    ax.xaxis.label.set_fontsize(25)
    ax.set_ylabel("b($^o$)")
    ax.yaxis.label.set_fontsize(25)
    ax.grid(True)
    return cs

def Dist_clust(data):
    """Calculate the distance in flux from each spectra to its cluster and the next nearst.
    Parameters
    -----------
    data: dictionary 
    Output of the k-means classification

    Retunrs:
    --------
    distance: 2D array, float
        Distances to cluster center and to the next closest center.
    min_dist_cl: 2D array, int
        Reference to cluster index in the distance array.
        """
    mask = np.loadtxt('/net/tarea/scratch/Rafael/phd/apogee/python/comb_SkyTel_mask.dat')
    masked = np.where(mask == 1)[0]
    spectra_list = data['fullset']
    clusters = data['clusters']
    clusters = clusters.transpose()
    distance = np.zeros((len(spectra_list), 2))
    min_dist_cl = np.zeros((data['nc'], 2))
    for j_cluster in range(data['nc']):
        dist_cluster= np.zeros((data['nc']))
        for i_cluster in range(data['nc']):
            dist_cluster[i_cluster] = np.nansum((clusters[j_cluster][masked] - clusters[i_cluster][masked])**2)**0.5
        min_dist_cl[j_cluster,0] = np.argmin(dist_cluster)
        dist_cluster[np.argmin(dist_cluster)] = dist_cluster[np.argmax(dist_cluster)]
    if (len(np.where(dist_cluster != 0)[0]) > 0):
                min_dist_cl[j_cluster,1] = np.argmin(dist_cluster[(dist_cluster != 0)])
    for i_spec, name in enumerate(spectra_list):
            vec_temp = np.load(name)
            for i_cluster, j_cluster in enumerate(min_dist_cl[data['assign'][i_spec]]):
                    distance[i_spec,i_cluster] = np.nansum((clusters[j_cluster][masked] - vec_temp['norm'][masked])**2)**0.5
            vec_temp.close()
    return distance, min_dist_cl

def GAP(ks, Wkbs, Wks, sk):
    do_ks = np.unique(np.array(ks)[np.where(ks < max(np.array(ks)))[0]])
    return np.asarray([Wkbs[np.where(ks == k)].mean() - Wks[np.where(ks == k)].mean() 
                -(Wkbs[np.where(ks == k+1)].mean()
                -Wks[np.where(ks == k+1)].mean() 
                -sk[np.where(ks == k+1)].mean()) for k in do_ks])


def DIFF(ks, SSE, n_features):
    do_ks = np.array(ks)[np.where(ks > min(np.array(ks)))[0]]
    return do_ks, [(((k-1)**(2./n_features))*SSE[ks.index(k-1)] - (k**(2./n_features))*SSE[ks.index(k)]) for k in do_ks]

def SSE_clust(data, transpose=True, do_mask=False):
    """Calculate the Sum of Squares Error index for each cluster in the classification.
    font: https://hlab.stanford.edu/brian/error_sum_of_squares.html
    Parameters
    ----------

    fileName: string
        Name of the file with the k-means output.

    Returns:
    --------
    
    SSE_CLUST: array, float
        Sum of Squares Error for each cluster in the classification.
    """    
    mask = np.loadtxt('/net/tarea/scratch/Rafael/phd/apogee/python/comb_SkyTel_mask.dat')
    masked = np.where(mask == 1)[0]
    SSE_CLUST = np.zeros(data['nc'])
    if (transpose): sclusters = data['sclusters'].transpose()
    else: sclusters = data['sclusters']
    for NC in range(data['nc']):
        if (do_mask == True): SSE_CLUST[NC] = np.nansum(sclusters[NC][masked]**2)
        else: SSE_CLUST[NC] = np.nansum(sclusters[NC]**2)
    return SSE_CLUST

def CDF(data, parms, class_number, var):
    """Cumulative Distribution Function: https://en.wikipedia.org/wiki/Cumulative_distribution_function
    
    Parameters:
    ----------
    data: dictionary 
        Output of the k-means classification

    parms: array (n_sample, n_features)
        21 standar parameters for stars in the sample. 
        ['Teff', 'LOGG', 'MH', 'CM', 'NM', 'aM', 'Al', 'Ca', 'C', 'Fe', 'K', 'Mg', 'Mn',
         'Na', 'Ni', 'N', 'O', 'Si', 'S', 'Ti', 'V']
     class_number: int
        Class to witch we want calculate CDF.
    var: str
        String to reffer the parameter we are intreasted with.
    """
    select = [(data['assign'] == class_number) & (parms[var] != -9999)]
    vec_value, base = np.histogram(abs(parms[var][select] - np.nanmean(parms[var][select])), bins=500)
    cumulative = np.cumsum(vec_value)
    plt.plot(base[:-1], cumulative/data['count'][class_number], ls='-', marker=None)


def select_classes(class_list, data):
    str_exec_save = "condition = ["
    for i in range(len(class_list)):
        str_exec_save += "(data['assign'] == %s)" % (class_list[i])
        if (i != len(class_list) -1): str_exec_save += ' | '
        else: str_exec_save += ']'
    exec(str_exec_save)
    return condition

def calc_ratio(file_list, parms, good_err = None, parms_list = None, calc_dist=None):
            good_ratio = np.zeros(len(file_list))
            for i, inFile in enumerate(file_list):
                        data = np.load(inFile)
                        mean_par, std_par, std2_par, norm_std, cluster_list, good_cluster, dist_clust, min_dist_cl, SSE = sig_by_class(data, parms, good_err, parms_list, calc_dist)
                        good_ratio[i] = (np.nansum(data['count'][good_cluster]))/np.nansum(data['count'])
            np.savez('./stat/' + inFile.split('/')[-1], mean_par=mean_par,
                            std_par=std_par,
                            std2_par=std2_par,
                            norm_std=norm_std,
                            cluster_list=cluster_list,
                            good_cluster=good_cluster,
                            dist_clust=dist_clust,
                            min_dist_cl=min_dist_cl,
                            SSE = SSE)

            return good_ratio

def sig_cut(vec):
    """Calculate one and 2-sigma by CDF."""
    fsig = (0.682689492137)
    f2sig = (0.954499736104)                                
    sig = np.nanpercentile(abs(vec - np.nanmean(vec)), 100*fsig)
    sig2 = np.nanpercentile(abs(vec - np.nanmean(vec)), 100*f2sig)
    return np.mean(vec), sig, sig2, np.std(vec)


def sig_by_class(data, parms, good_err = None, parms_list = None, calc_dist=None):
    if (good_err == None): good_err = [350, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
            0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35]
    cluster_list = np.asarray(range(data['nc']))
    if (parms_list == None): parms_list = parms.dtype.names
    mean_par = np.zeros((len(cluster_list), len(parms_list)))
    std_par  = np.zeros((len(cluster_list), len(parms_list)))
    std2_par = np.zeros((len(cluster_list), len(parms_list)))
    norm_std = np.zeros((len(cluster_list), len(parms_list)))
    for i, n in enumerate(cluster_list):
        for j, par in enumerate(parms_list):
            clean = [(parms[par] != -9999) & (data['assign'] == n)]
            if (np.sum(clean) != 0):
                mean_par[i,j], std_par[i,j], std2_par[i,j], norm_std[i,j] = sig_cut(parms[par][clean])
            else:
                mean_par[i,j], std_par[i,j], std2_par[i,j], norm_std[i,j] = 4*[np.nan]
    if (calc_dist != None): dist_clust, min_dist_cl = Dist_clust(data)
    else: dist_clust = np.ones((len(data['assign']), 2)); min_dist_cl = np.zeros((len(data['assign']), 2))
    good_cluster = []
    for k, nc in enumerate(cluster_list):
            if(np.sum(std_par[k,:] <= good_err ) == len(good_err) 
            and data['count'][nc] > 10 
            and dist_clust[:,0][(data['assign'] == nc)].mean() < 2.5): good_cluster.append(nc)
    SSE = SSE_clust(data)
    return mean_par, std_par, std2_par, norm_std, cluster_list, good_cluster, dist_clust, min_dist_cl, SSE

def plot_good(fileName, parms, leg=None, labels=None, NC=None, parms_list = None, good_err=None):
      data = np.load(fileName)
      mean_par, std_par, std2_par, norm_std, cluster_list, good_cluster, dist_clust, min_dist_cl, SSE = sig_by_class(data, parms, good_err, parms_list)
      plot_errorbar(fileName, parms, mean_par, std_par, good_cluster)
      single_parms(fileName, parms, leg, labels, NC=good_cluster)
      return mean_par, std_par, std2_par, norm_std, cluster_list, good_cluster, dist_clust, min_dist_cl

def plot_errorbar(fileName, parms, mean_par, std_par, good_cluster):
    data = np.load(fileName)
    fig = plt.figure(figsize=(22,12))
    ax = fig.add_subplot(1,2,1)
    scatter_kwargs = {"zorder":100}
    error_kwargs = {"lw":.5, "zorder":0}
    non_zero = np.where(data['count'] > 0)[0]
    non_zero = np.where(data['count'] > 0)[0]
    plt.scatter(mean_par[non_zero,0], mean_par[non_zero,1], c=np.log10(data['count'][non_zero]), 
            marker='o', s=30,  vmin=1.0, vmax = 4.0, cmap= cm.hot_r, **scatter_kwargs)
    plt.errorbar(mean_par[non_zero,0], mean_par[non_zero,1], xerr=std_par[non_zero,0], yerr=std_par[non_zero,1], fmt=None, marker=None, mew=0,**error_kwargs) 
    plt.ylabel('log(g) (dex)', fontsize=28)
    plt.xlabel('Teff (k)', fontsize=28)
    plt.xlim(8100,3400)
    plt.ylim(5.05,-0.05)
    ax = fig.add_subplot(1,2,2)
    scatter_kwargs = {"zorder":100}
    error_kwargs = {"lw":.5, "zorder":0}
    plt.scatter(mean_par[good_cluster,0], mean_par[good_cluster,1], c=np.log10(data['count'][good_cluster]), 
            marker='o', s=30,  vmin=1.0, vmax = 4.0, cmap=cm.hot_r, **scatter_kwargs)
    plt.errorbar(mean_par[good_cluster,0], mean_par[good_cluster,1], xerr=std_par[good_cluster,0], yerr=std_par[good_cluster,1], fmt=None, marker=None, mew=0,**error_kwargs) 
    cbar = plt.colorbar()
    cbar.set_label('log(N)', fontsize=28)
    plt.ylabel('log(g) (dex)', fontsize=28)
    plt.xlabel('Teff (k)', fontsize=28)
    plt.xlim(8100,3400)
    plt.ylim(5.05,-0.05)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig = plt.figure(figsize=(22,12))
    ax = fig.add_subplot(1,2,1)
    scatter_kwargs = {"zorder":100}
    error_kwargs = {"lw":.5, "zorder":0}
    non_zero = np.where(data['count'] > 0)[0]
    non_zero = np.where(data['count'] > 0)[0]
    plt.scatter(mean_par[non_zero,2], mean_par[non_zero,5], c=np.log10(data['count'][non_zero]), 
            marker='o', s=30,  vmin=1.0, vmax = 4.0, cmap=cm.hot_r, **scatter_kwargs)
    plt.errorbar(mean_par[non_zero,2], mean_par[non_zero,5], xerr=std_par[non_zero,2], yerr=std_par[non_zero,5], fmt=None, marker=None, mew=0,**error_kwargs) 
    plt.ylabel('[alpha/M] (dex)', fontsize=28)
    plt.xlabel('[M/H] (dex)', fontsize=28)
    ax = fig.add_subplot(1,2,2)
    scatter_kwargs = {"zorder":100}
    error_kwargs = {"lw":.5, "zorder":0}
    plt.scatter(mean_par[good_cluster,2], mean_par[good_cluster,5], c=np.log10(data['count'][good_cluster]), 
            marker='o', s=30,  vmin=1.0, vmax = 4.0, cmap=cm.hot_r, **scatter_kwargs)
    plt.errorbar(mean_par[good_cluster,2], mean_par[good_cluster,5], xerr=std_par[good_cluster,2], yerr=std_par[good_cluster,5], fmt=None, marker=None, mew=0,**error_kwargs) 
    cbar = plt.colorbar()
    cbar.set_label('log(N)', fontsize=28)
    plt.ylabel('[alpha/M] (dex)', fontsize=28)
    plt.xlabel('[M/H] (dex)', fontsize=28)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig = plt.figure(figsize=(12,12))
    plt.bar(range(data['nc']),
        np.log10(data['count'][:data['nc']]),
        label=str(int(np.nansum(data['count'])))+' stars in the full sample.')
    plt.bar(good_cluster,
        np.log10(data['count'][good_cluster]),
        label=str(round(100*np.nansum(data['count'][good_cluster])/np.nansum(data['count']),2))+'% of the stars in the good sample.',
        color='r')
    plt.xlabel('Class ID', fontsize=18)
    plt.ylabel('log(N)', fontsize=18)
    plt.xlim(0,200)
    plt.legend()
    plt.tight_layout()
    data.close()

def plot_class(fileName, parms, leg=None, labels=None, NC=None, clusters=None, outFile=None, Group=0):
    mask = np.loadtxt('/net/tarea/scratch/Rafael/phd/apogee/python/comb_SkyTel_mask.dat')
    wavelenght = np.load('/scratch/Rafael/phd/apogee/python/wavelenght.npz')
    mask_inv = np.zeros(len(mask))
    x = np.ones(len(mask))*range(len(mask))
    y0 = np.zeros(len(mask))
    y1 = np.ones(len(mask))*1.2
    for i_mask, j_mask in enumerate(mask):
        if (j_mask == 0): mask_inv[i_mask] = 1
    data = np.load(fileName)
    if (clusters==None): clusters = data['clusters']
    mk = ['x', '^', 's', '>', '.', '<', 'x', 'v', 'o', 'p', 'h']
    COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    CK = np.array(list(itertools.product(mk,COLORS)))
    mk, COLORS = CK[:,0], CK[:,1]
    if (NC == None): NC = range(data['nc'])
    if (labels == None): 
        labels = parms.dtype.names
        leg = ['$Teff\,(K)$', '$\\log(g)\,(dex)$', '$[M/H]\,(dex)$', '$[C/M]\,(dex)$', '$[N/M]\,(dex)$', '$[\\alpha/M]\,(dex)$']
        for P in parms.dtype.names[6:]: leg += ['$['+ P + '/H]\,(dex)$']
    Max = [parms[P][(parms[P] != -9999)].max() for P in parms.dtype.names[1:]]
    Min = [parms[P][(parms[P] != -9999)].min() for P in parms.dtype.names[1:]]
    #x = np.arange(1.514, 1.696, (1.696-1.514)/clusters[0].shape[0])
    x = wavelenght['AIR']
    for i_class in NC:
        fig = plt.figure(figsize=(21,11.0))
        ax = plt.subplot2grid((5,1), (0, 0), rowspan=1)
        clean = [(parms[labels[0]] != -9999) & (data['assign'] == i_class)]
        MEAN_T = round(np.nanmedian(parms[labels[0]][clean]),1)
        try: STD_T = round(np.percentile(abs(parms[labels[0]][clean]-np.nanmedian(parms[labels[0]][clean])), 68.2689492),1)
        except: STD_T = 0
        plt.xlabel('$\\lambda \,(\\AA )$'); plt.ylabel('$Flux$', fontsize=20)
        #plt.plot(x, clusters[i_class], color=COLORS[i_class], ls='-', label="$Class = " + str(i_class) + '$\n$N_{\\star} = ' + str(data['count'][i_class])+'$' + '\n $Teff = ' + str(MEAN_T) + ' \\pm ' + str(STD_T)+ '$')
        plt.plot(x, clusters[i_class], color=COLORS[i_class], ls='-', 
            label=("$G%d $\n $Class = %02d $ \n $N_{\\star} = %d $ \n $Teff = %4.1f \\pm %4.1f $" % (Group, i_class, data['count'][i_class], MEAN_T, STD_T)))
        plt.legend(loc=3)
        plt.fill_between(x,y0,y1,where=mask_inv, color='k', alpha=0.3)
        plt.ylim(0.6,1.1)
        plt.tick_params(axis='y', which='both', labelsize=20)
        plt.yticks(np.arange(0.7, 1.1, 0.2))
        plt.xticks(np.arange(15000, 17000, 125))
        plt.grid(True)
        for i_L, L in enumerate(labels[1:]):
            ax = plt.subplot2grid((5,5), ((i_L/5) +1, i_L%5), rowspan=1)
            clean = [(parms[labels[0]] != -9999) & (parms[L]!= -9999) & (data['assign'] == i_class)]
            MEAN = np.nanmedian(parms[L][clean])
            MEAN_T = round(np.nanmedian(parms[labels[0]][clean]),1)
            try: STD = np.percentile(abs(parms[L][clean]-np.nanmedian(parms[L][clean])), 68.2689492)
            except: STD = 0
            LAB = '$\\langle x \\rangle = ' + str(round(MEAN,2)) + ' \\pm ' + str(round(STD,2))+'$'
            clean0 = [(parms[labels[0]] != -9999) & (parms[L]!= -9999)]
            RANGE0 = [(min(parms[labels[0]][clean0]), max(parms[labels[0]][clean0])), (min(parms[L][clean0]), max(parms[L][clean0]))]
            plt.hist2d(parms[labels[0]][clean0], parms[L][clean0], bins=(50,50), cmin=100, cmap=cm.cool, range=RANGE0);
            plt.hist2d(parms[labels[0]][clean], parms[L][clean], bins=(50,50), cmin=data['count']/2500., cmap=cm.autumn, range=RANGE0)
            plt.plot(MEAN_T, MEAN, 'rx', label=LAB)
            plt.grid(True)
            cbar = plt.colorbar(aspect= 10)
            plt.ylabel(leg[1:][i_L], fontsize=20)
            plt.xticks(np.arange(8000,3000,-1500)); plt.yticks(np.arange(round(Min[i_L],1), round(Max[i_L],1), 0.5))
            plt.tick_params(axis='y', which='both', labelsize=15)
            if (i_L == 0): plt.xlim(8500, 3400); plt.ylim(Max[i_L], Min[i_L]) 
            else: plt.xlim(8500, 3400); plt.ylim(Min[i_L], Max[i_L])
            if (i_L == 17): plt.xlabel('$T_{eff}\, (K)$', fontsize=20)
            if(i_L > 14): 
                plt.tick_params(axis='x', which='both', labelsize=15)
                plt.legend(bbox_to_anchor=(1.00, -0.15))
            else:
                plt.tick_params(axis='x', which='both', labelsize=0)
                plt.legend(bbox_to_anchor=(1.00, 0.15))
        plt.tight_layout(); fig.subplots_adjust(hspace=0.1)
        if (outFile != None): 
            plt.savefig('%s_%02d.pdf' % (outFile, i_class))
            plt.clf(); plt.close()
        else: plt.show()
            


def single_parms(fileName, parms, leg=None, labels=None, NC=None, alpha=0.1, TITLE=None):
    mk = ['x', '^', 's', '>', '.', '<', 'x', 'v', 'o', 'p', 'h']
    COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    CK = np.array(list(itertools.product(mk,COLORS)))
    mk, COLORS = CK[:,0], CK[:,1]
    if (labels == None): 
        labels = [parms.dtype.names[0], parms.dtype.names[1], parms.dtype.names[2], parms.dtype.names[3]]
        leg = ['$Teff\,(K)$', '$\\log(g)\,(dex)$', '$[M/H]\,(dex)$', '$[C/M]\,(dex)$', '$[N/M]\,(dex)$', '$[\\alpha/M]\,(dex)$']
    for P in parms.dtype.names[6:]: leg += ['$['+ P + '/H]\,(dex)$']
    data = np.load(fileName)
    if (NC == None): NC = range(data['nc'])
    for i_L, L in enumerate(labels[1:]):
        fig = plt.figure(figsize=(14,12))
        for i_split in range(6):
            ax = fig.add_subplot(2,3,i_split+1)
            for i_nc in NC[(i_split*(data['nc']/7)):(data['nc']/7)*(i_split+1)]:
                clean1 = [(parms[labels[0]] != -9999) & (parms[L]!= -9999) & (data['assign'] == i_nc)]
                plt.plot(parms[labels[0]][clean1], parms[L][clean1], ''.join((COLORS[i_nc],mk[i_nc])), alpha=alpha) 
                if(labels[0] == 'Teff'): plt.xlim(8100,3400)
                if(L == 'LOGG'): plt.ylim(5.05,-0.05)
                if(TITLE!=None): plt.title(TITLE, fontsize=25)
                plt.xlabel(leg[0], fontsize=25)
                plt.ylabel(leg[i_L+1], fontsize=25)
                plt.tight_layout()
        plt.show()
########for i_nc in NC:
########        clean1 = [(parms['LOGG'] != -9999) & (parms['Teff'] != -9999) & (data['assign'] == i_nc)]
########        plt.plot(parms['Teff'][clean1], parms['LOGG'][clean1], ''.join((COLORS[i_nc],mk[i_nc])), alpha=alpha)
########        plt.xlim(8100,3400);
########        plt.ylim(5.05,-0.05)
########        ratio = ' - ' + str(round(100*np.nansum(data['count'][NC])/np.nansum(data['count']), 2)) + '% of the stars in this plot.'
########        plt.title('NC = '+str(data['nc'])+ratio, fontsize=25)
########        plt.xlabel('$Teff (K)$', fontsize=25)
########        plt.ylabel(''.join(('$\\log(g) (dex)$')), fontsize=25)
########        plt.tight_layout()
########plt.show()
########fig = plt.figure(figsize=(14,12))
########data = np.load(f)
########if (NC == None): NC = range(data['nc'])
########for i_nc in NC:
########        clean1 = [(parms['MH'] != -9999) & (parms['Teff'] != -9999) & (data['assign'] == i_nc)]
########        plt.plot(parms['Teff'][clean1], parms['MH'][clean1], ''.join((COLORS[i_nc],mk[i_nc])), alpha=alpha)
########        plt.xlim(8100,3400)
########        plt.ylim(-2.55,0.55)
########        ratio = ' - ' + str(round(100*np.nansum(data['count'][NC])/np.nansum(data['count']), 2)) + '% of the stars in this plot.'
########        plt.title('NC = '+str(data['nc'])+ratio, fontsize=25)
########        plt.xlabel('$Teff (K)$', fontsize=25)
########        plt.ylabel(''.join(('$[M/H] (dex)$')), fontsize=25)
########        plt.tight_layout()
########plt.show()
########for j_label, label in enumerate(labels):
########        fig = plt.figure(figsize=(14,12))
########        data = np.load(f)
########        if (NC == None): NC = range(data['nc'])
########        for i_nc in NC:
########                clean1 = [(parms[label] != -9999) & (parms['MH'] != -9999) & (data['assign'] == i_nc)]
########                plt.plot(parms['MH'][clean1], parms[label][clean1], ''.join((COLORS[i_nc],mk[i_nc])), alpha=alpha)
########                plt.xlim(-2.55,0.55);
########                ratio = ' - ' + str(round(100*np.nansum(data['count'][NC])/np.nansum(data['count']), 2)) + '% of the stars in this plot.'
########                plt.title('NC = '+str(data['nc'])+ratio, fontsize=25)
########                plt.xlabel('$[M/H] (dex)$', fontsize=25)
########                plt.ylabel(''.join(('',leg[j_label],' (dex)')), fontsize=25)
########                plt.tight_layout()
########        plt.show()

def compare_subparms(f1,f2, parms, subparms, leg=None, labels=None, NC=None):
        mk = 5*['.', '.', '.', '.', '.', '.', '^', '^', '^', '^', '^', '^', 's', 's', 's', 's', 's', 's', '>', '>', '>', '>', '>', '>', '<', '<', '<', '<', '<', '<', 'v', 'v', 'v', 'v', 'v', 'v']
        COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']*25
        if (labels == None): labels = parms.dtype.names[1:6]
        if (leg == None): leg = ['log(g)', '[M/H]', '[C/M]', '[N/M]', '[alpha/M]']
        for j, label in enumerate(labels):
                fig = plt.figure(figsize=(22,12))
                data1 = np.load(f1)
                ax = fig.add_subplot(1,2,1)
                SSE = mypl.SSE_clust(data1)
                NC = np.where((SSE > np.median(SSE)))[0]
                for i in NC:
                        clean1 = [(parms[label] != -9999) & (parms['Teff'] != -9999) & (data1['assign'] == i)]
                        plt.plot(parms['Teff'][clean1], parms[label][clean1], ''.join((COLORS[i],mk[i])))
                        plt.xlim(8100,3400);
                        if (j == 0): plt.ylim(5.05,-0.05)
                        plt.title('Classes - NC = '+str(data1['nc']), fontsize=25)
                        plt.xlabel('Teff (K)', fontsize=25)
                        plt.ylabel(''.join(('',leg[j],' (dex)')), fontsize=25)
                ax = fig.add_subplot(1,2,2)
                data2 = np.load(f2)
                NC = range(data2['nc'])
                for i in NC:
                        clean2 = [(subparms[label] != -9999) & (subparms['Teff'] != -9999) & (data2['assign'] == i)]
                        plt.plot(subparms['Teff'][clean2], subparms[label][clean2], ''.join((COLORS[i],mk[i])))
                        plt.xlim(8100,3400);
                        plt.title('Subclasses - NC = '+str(data2['nc']), fontsize=25)
                        plt.xlabel('Teff (K)', fontsize=25)
                        plt.ylabel(''.join(('',leg[j],' (dex)')), fontsize=25)
                        if (j == 0): plt.ylim(5.05,-0.05)
                plt.tight_layout()
        plt.show()



def compare_parms(f1,f2, parms, leg=None, labels=None, NC=None):
    mk = 5*['.', '.', '.', '.', '.', '.', '^', '^', '^', '^', '^', '^', 's', 's', 's', 's', 's', 's', '>', '>', '>', '>', '>', '>', '<', '<', '<', '<', '<', '<', 'v', 'v', 'v', 'v', 'v', 'v']
    COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']*25
    if (labels == None): labels = parms.dtype.names[1:6]
    if (leg == None): leg = ['log(g)', '[M/H]', '[C/M]', '[N/M]', '[alpha/M]']
    for j, label in enumerate(labels):
         fig = plt.figure(figsize=(17,8))
         data1 = np.load(f1)
         ax = fig.add_subplot(1,2,1)
    if (NC == None): NC = range(data1['nc'])
    for i in NC:
            clean1 = [(parms[label] != -9999) & (parms['Teff'] != -9999) & (data1['assign'] == i)]
            plt.plot(parms['Teff'][clean1], parms[label][clean1], ''.join((COLORS[i],mk[i])), alpha=0.07)
            plt.xlim(8100,3400);
            if (j == 0): plt.ylim(5.05,-0.05)
            plt.title('Synthetic - NC = 69', fontsize=25)
            plt.xlabel('Teff (K)', fontsize=25)
            plt.ylabel(''.join(('',leg[j],' (dex)')), fontsize=25)
    ax = fig.add_subplot(1,2,2)
    data2 = np.load(f2)
    if (NC == None): NC = range(data2['nc'])
    for i in NC:
            clean2 = [(parms[label] != -9999) & (parms['Teff'] != -9999) & (data2['assign'] == i)]
            plt.plot(parms['Teff'][clean2], parms[label][clean2], ''.join((COLORS[i],mk[i])), alpha=0.07)
            plt.xlim(8100,3400);
            plt.title('Synthetic 1line - NC = 69', fontsize=25)
            plt.xlabel('Teff (K)', fontsize=25)
            plt.ylabel(''.join(('',leg[j],' (dex)')), fontsize=25)
            if (j == 0): plt.ylim(5.05,-0.05)
    plt.show()


def write_coo(RA, DEC, fileName):
    out = open(fileName, 'w')
    out.write(
'''|              ra|              dec|
|          double|           double|
|             deg|              deg|
'''
        )
    for i in range(len(RA)):
        out.write('%17.12f %17.13f\n' % (RA[i], DEC[i]))
    out.close()    

def distances(data):
    spectra_list = data['fullset']
    clusters = data['clusters']
    clusters = clusters.transpose()
    distance = np.zeros((len(spectra_list), data['nc']))
    for i_spec, name in enumerate(spectra_list):
        vec_temp = np.load(name)
        for j_cluster in xrange(data['nc']):
            distance[i_spec,j_cluster] = np.nansum((clusters[j_cluster] - vec_temp['norm'])**2)
            distance[i_spec,j_cluster] = distance[i_spec, j_cluster]**0.5
        vec_temp.close()
    return distance

def get_class(assign, data, ID_CLASS):
    data = data[np.where(assign == ID_CLASS)[0]]
    for i in range(len(data[0])): 
        data = data[np.where(data[:,i] != -9999)]
    return data

def get_parms(ID_LIST, key_words=0, positions=0, refs=0, font='FPARAM'):
    if (key_words==0):
        key_words = []
        for i in range(6): key_words.append(font)
        positions = [0, 1, 3, 4, 5, 6] 
        refs = ['Teff', 'LOGG', 'MH', 'CM', 'NM', 'aM']
        EL = ['Al', 'Ca', 'C', 'Fe', 'K', 'Mg', 'Mn', 'Na', 
            'Ni', 'N', 'O', 'Si', 'S', 'Ti', 'V']
        for i in range(15): 
            positions.append(i)
            key_words.append('FELEM')
            refs.append(EL[i])
    suffix = '_parm.npz'
    data = []
    for name in ID_LIST:
        vec_temp = np.load(''.join([name.split('_')[0], suffix]))
        line = []
        for i, word in enumerate(key_words):
            try: line.append(vec_temp[word][positions[i]])
            except IndexError: line.append(vec_temp[word])
        data.append(line)
    data = np.rec.fromrecords(data, names=refs)
    return data


def stdpl(x, y=None, sx=None, sy=None, title='', xlabel='', ylabel='', color='r', marker='o',
        mksize=8, label='', figname=None, xlim=[None,None], ylim=[None,None], ls='',
        width=35, height=25, xticks=20, yticks=20, legendsize=20, hist=None):
    fig = plt.figure(figsize=(width*0.393701, height*0.393701))
    ax = fig.add_subplot(111)
    plt.tick_params(axis='x', which='both', labelsize=xticks)
    plt.tick_params(axis='y', which='both', labelsize=yticks)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    if(xlim[0]!=None and xlim[1]!=None):
        ax.set_xlim(xlim[0],xlim[1])
    else:
        xlim=[np.mean(x)-4*np.std(x), np.mean(x)+4*np.std(x)] 
    if(ylim[0]!=None and ylim[1]!=None):
        ax.set_ylim(ylim[0],ylim[1])
    else:
        ylim=[np.mean(y)-4*np.std(y), np.mean(y)+4*np.std(y)]
    if (hist!=None or y==None):
        plt.hist(x, label=label, color=color)
    else:
        plt.errorbar(x, y, sx, sy, label=label, color=color, ls=ls, marker=marker, markersize=mksize)
    ax.legend(bbox_to_anchor=(1.1, 1.1), prop={'size':legendsize})
    if (figname==None):
        plt.show()
    else:
        plt.savefig(figname + ".pdf")

#   def find_levels(H, N_CLASS):
#           levels = np.zeros(5)
#           xmax = np.zeros(len(H))
#           for j in range(len(H)):
#                   xmax[j] = max(H[j])
#           i_init = np.argmax(xmax)
#           j_init = np.argmax(H[i_init])
#           levels[4] = max(xmax)
#           PORC_LEVELS= [0.68268,0.45,0.30,0.15]
#           for i_level, PORCENTAGE in enumerate(PORC_LEVELS):
#                   suma = 0; k = 0
#                   while(suma < PORCENTAGE*N_CLASS):
#                           suma = np.sum(H[np.where(H >= H[i_init,j_init] - k)])
#                           k +=1
#                   levels[i_level] = H[i_init,j_init] - k
#           if (levels[0] < 1): levels[1] = 1
#           for i in range(4):
#                   if(levels[i] >= levels[i+1]): levels[i+1] += 0.01
#           return levels
#def find_levels(H, N_CLASS):
#   h = H.flatten()
#   S = []
#   hs = np.sort(h)
#   for i_value in range(len(hs)): S.append(hs[:i_value][::-1].sum())
#   S = np.array(S)
#   levels = []
#   for ratio in [0.6827,0.45,0.30,0.15][::-1]: levels.append(hs[np.argmin(abs(S - ratio*S[-1]))])
#   return np.array(levels)

def find_levels(H, N_CLASS):
    h = H.flatten()
    S = []
    hs = np.sort(h)
    for i_value in range(len(hs)): S.append(hs[:i_value][::-1].sum())
    S = np.array(S)
    levels = []
    for ratio in [0.6827,0.45,0.30,0.15][::-1]:
        levels.append(hs[np.argmin(abs(S - ratio*S[-1]))])
    levels = np.array(levels)
    for i in range(3):
        if levels[i+1] <= levels[i]:
            return [0]
    return levels


def plot_countour(Lx, Ly, assign, fparms, CLASSES=range(10)):
    ml = ['-', '--', ':', '--', '-', ':', '-', '--']
    CMAP = ['Blues', 'Reds', 'Greys', 'Purples', 'Greens', 'Oranges', 'pink_r', 'YlGn']
    CK = np.array(list(itertools.product(ml,CMAP)))
    Px = fparms.dtype.names.index(Lx)
    Py = fparms.dtype.names.index(Ly)
    C = 0
    fig = plt.subplots(figsize=(12,8.0))
    index_color = 0
    for i in CLASSES:
        filt = [(fparms[Lx] != -9999) & (fparms[Ly] != -9999)]
        MIN_X = min(fparms[Lx][filt])
        MIN_Y = min(fparms[Ly][filt])
        MAX_X = max(fparms[Lx][filt])
        MAX_Y = max(fparms[Ly][filt])
        filt = [(assign == i) & (fparms[Lx] != -9999) & (fparms[Ly] != -9999)]
        H, xedges, yedges = np.histogram2d(fparms[Lx][filt], fparms[Ly][filt], bins=(50,50))
        extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
        N_CLASS = len(fparms[Ly][filt])
        levels = np.array(find_levels(H, N_CLASS))
        plt.contourf(H.transpose(), levels, extent=[xedges.min(), xedges.max(), yedges.min(),yedges.max()], cmap=CK[:,1][index_color], alpha=0.7, vmin=1)    
        index_color +=1
        plt.text(np.nanmedian(fparms[Lx][filt]), np.nanmedian(fparms[Ly][filt]), '$C%02d$'%(i), fontsize=20)
    plt.xlabel('$%s$'%Lx, fontsize=20);
    plt.ylabel('$%s$'%Ly, fontsize=20);
    plt.xlim(MIN_X, MAX_X)
    plt.ylim(MIN_Y, MAX_Y)
    plt.show()


def hist_plot(DATA, nbins):
    clusters = np.load('clusters_full.npy')
    plt.title('Histogram of classes', fontsize=20)
    plt.xlabel('Class number ID', fontsize=20)
    plt.ylabel('Number of stars in class', fontsize=20)
    plt.hist(DATA, bins=nbins); plt.show()

def spectra_pile(clusters, count):
    mask = np.loadtxt('/net/tarea/scratch/Rafael/phd/apogee/python/comb_SkyTel_mask.dat')
    mask_inv = np.zeros(len(mask))
    x = np.ones(len(mask))*range(len(mask))
    y0 = np.zeros(len(mask))
    y1 = np.ones(len(mask))*22
    for i_mask, j_mask in enumerate(mask):
        if (j_mask == 0): mask_inv[i_mask] = 1
    control = 0
    j = 0
    while  (control <= len(clusters[0])):
        plt.title('Classes mean spectrum', fontsize=20)
        plt.xlabel('Nth pixel', fontsize=20)
        plt.ylabel('Norm. flux + class number', fontsize=20)
        plt.xlim(0,9450) 
        plt.ylim(0,22)
        plt.xlabel('Pixel') 
        plt.ylabel('Norm. Flux')
        for i in range(len(count)/20): 
            try:
                plt.plot(clusters[:,i+10*j]+(20-i), label=int(count[i+10*j]))
                control += 1
            except IndexError:
                control += 1
        j += 1
        plt.fill_between(x,y0,y1,where=mask_inv, color='k', alpha=0.3)
        plt.legend()
        plt.show()

def gif(data, data_stars, fileRoot):
    mask = np.loadtxt('/net/tarea/scratch/Rafael/phd/apogee/python/comb_SkyTel_mask.dat')
    mask_inv = np.zeros(len(mask))
    x = np.ones(len(mask))*range(len(mask))
    y0 = np.zeros(len(mask))
    y1 = np.ones(len(mask))*1.2
    for i_mask, j_mask in enumerate(mask):
        if (j_mask == 0): mask_inv[i_mask] = 1
    assign = data['assign']
    count = data['count']
    count = count[np.where(count != 0)]
    clusters = data['clusters']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']*25
    clean = [(data_stars['LOGG'] != -9999) & (data_stars['LOGG'] != 9999)]
    MIMLOGG, MAXLOGG = min(data_stars['LOGG'][clean]), max(data_stars['LOGG'][clean])
    clean = [(data_stars['MH'] != -9999) & (data_stars['MH'] != 9999)]
    MIMMC, MAXMC = min(data_stars['MH'][clean]), max(data_stars['MH'][clean])
    clean = [(data_stars['MH'] != -9999) & (data_stars['MH'] != 9999)]
    MIMCM, MAXCM = min(data_stars['CM'][clean]), max(data_stars['CM'][clean])
    clean = [(data_stars['NM'] != -9999) & (data_stars['NM'] != 9999)]
    MIMNM, MAXNM = min(data_stars['NM'][clean]), max(data_stars['NM'][clean])
    clean = [(data_stars['aM'] != -9999) & (data_stars['aM'] != 9999)]
    MIMalphaM, MAXalphaM = min(data_stars['aM'][clean]), max(data_stars['aM'][clean])
    for i in range(len(count)):
        fig = plt.figure(figsize=[18,10])
        lista = np.where(assign == i)[0]
        ax = fig.add_subplot(2,1,1); ax.set_xlabel('Pixel'); ax.set_ylabel('Norm. Flux'); ax.set_title('Class ' + str(i) + ' - ' + str(int(count[i])) + ' Members') 
        plt.plot(clusters[:,i], ls='-', color=colors[i]); ax.set_ylim(0,1.2)
        plt.fill_between(x,y0,y1,where=mask_inv, color='k', alpha=0.3)

        ax = fig.add_subplot(2,5,6); ax.set_xlabel('Teff (K)'); ax.set_ylabel('$\log(g) (dex)$')
        plt.xlim(9000,3000); ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e')); ax.xaxis.set_ticks([9000,6000,3000])
        plt.plot(data_stars['Teff'][lista], data_stars['LOGG'][lista], ls='', marker='.', markersize=0.2, color=colors[i]); ax.set_ylim(MAXLOGG, MIMLOGG)
        contour_plot(data_stars['Teff'][lista], data_stars['LOGG'][lista])

        ax = fig.add_subplot(2,5,7); ax.set_xlabel('Teff (K)'); ax.set_ylabel('$[M/H] (dex)$')
        plt.xlim(9000,3000); ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e')); ax.xaxis.set_ticks([9000,6000,3000])
        plt.plot(data_stars['Teff'][lista], data_stars['MH'][lista], ls='', marker='.', markersize=0.2, color=colors[i]); ax.set_ylim(MIMMC, MAXMC)
        contour_plot(data_stars['Teff'][lista], data_stars['MH'][lista])    
    
        ax = fig.add_subplot(2,5,8); ax.set_xlabel('Teff (K)'); ax.set_ylabel('$[C/M] (dex)$')
        plt.xlim(9000,3000); ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e')); ax.xaxis.set_ticks([9000,6000,3000])
        plt.plot(data_stars['Teff'][lista], data_stars['CM'][lista], ls='', marker='.', markersize=0.2, color=colors[i]); ax.set_ylim(MIMCM, MAXCM)
        contour_plot(data_stars['Teff'][lista], data_stars['CM'][lista])

        ax = fig.add_subplot(2,5,9); ax.set_xlabel('Teff (K)'); ax.set_ylabel('$[N/M] (dex)$')
        plt.xlim(9000,3000); ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e')); ax.xaxis.set_ticks([9000,6000,3000])
        plt.plot(data_stars['Teff'][lista], data_stars['NM'][lista], ls='', marker='.', markersize=0.2, color=colors[i]); ax.set_ylim(MIMNM, MAXNM)
        contour_plot(data_stars['Teff'][lista], data_stars['NM'][lista])

        ax = fig.add_subplot(2,5,10); ax.set_xlabel('Teff (K)'); ax.set_ylabel('$[alpha/M] (dex)$')
        plt.xlim(9000,3000); ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e')); ax.xaxis.set_ticks([9000,6000,3000])
        plt.plot(data_stars['Teff'][lista], data_stars['aM'][lista], ls='', marker='.', markersize=0.2, color=colors[i]); ax.set_ylim(MIMalphaM, MAXalphaM)
        contour_plot(data_stars['Teff'][lista], data_stars['aM'][lista])
    
        fig.subplots_adjust(wspace=0.5)
        plt.savefig('%s-%02d.png' % (fileRoot, i))
        plt.clf()
        plt.close()
    command = ('convert -delay 100 -loop 0 %s* %s.gif' % (fileRoot, fileRoot))
    os.system(command)

def tex_images(image_list, outFile='figures', FRAME='', BLOCK=''):
    out = open(outFile + '.tex', 'w')
    for image in image_list:
        FRAME_old, BLOCK_old = FRAME, BLOCK
        print(image[:-1])
        FRAME = raw_input('Frame title (%s): ' % FRAME)
        if (FRAME == ''): FRAME = FRAME_old
        BLOCK = raw_input('Block title (%s): ' % BLOCK)
        if (BLOCK == ''): BLOCK = BLOCK_old
        out.write('''
\\begin{frame}
\t\\frametitle{%s}
\t\\vspace{-0.3cm}
\t\\begin{minipage}{0.9\\textwidth}
\t\\begin{block}{%s}
\t\t\\begin{figure}[H]
\t\t\\centering
\t\t\\includegraphics[width=1.0\\textwidth]{%s}
\t\t\\end{figure}
\t\\end{block}
\t\\end{minipage}
\\end{frame}
''' % (FRAME, BLOCK, image[:-1]))
    out.close()

def tex_table(array, LABELS, counts, outFile, label='', Q_UP=np.zeros(1), Q_DOWN=np.zeros(1)):
    out = open(outFile, 'w')
    separ = ''
    for i in range(array.shape[1] + 2): separ += 'c'
    out.write('\\renewcommand{\\arraystretch}{1.05}\n\\begin{table}\n\t\\centering\n\t\\caption{\\label{tab:%s}}\n\t\\begin{tabular}{%s}\n' % (label,separ))
    out.write('Class & ')
    for name in LABELS:
        out.write('%s & ' % name)
    out.write('$N_{\\star}$ \\\\ \\hline \n')
    for i in range(array.shape[0]):
        out.write('$%2d$ & ' % i)
        for j in range(array.shape[1]):
            if (Q_UP.shape == array.shape):
                out.write('$%+-7.2f \\pm^{%8.2f}_{%8.2f}$ & ' % (array[i,j], Q_UP[i,j], Q_DOWN[i,j]))
            else:
                out.write('$%+-7.2f$ & ' % array[i,j])
        out.write('$%5d$ \\\\ \n' % counts[i])
    out.write('\\end{tabular}\n\\end{table}')
    out.close()

def quantiles_calc(parms, assign):
    K = len(np.unique(assign))
    VARS = parms.dtype.names
    n_vars = len(parms.dtype.names)
    QUANTILES = {}
    assign = np.copy(assign)
    for k in range(K):
        for var in VARS:
            filt = [(assign == k) & (parms[var] != -9999)]
            if(sum(filt[0]) > 0):
                QUANTILES['K%d-%s-median'%(k,var)] = np.nanmedian(parms[var][filt])
                QUANTILES['K%d-%s-q-'%(k,var)] = np.percentile(parms[var][filt],50.-34.1344746)
                QUANTILES['K%d-%s-q+'%(k,var)] = np.percentile(parms[var][filt],50.+34.1344746)
            else:
                QUANTILES['K%d-%s-median'%(k,var)] = 0
                QUANTILES['K%d-%s-q-'%(k,var)] = 0
                QUANTILES['K%d-%s-q+'%(k,var)] = 0
    for var in VARS:
        exec('%s_median = np.array([QUANTILES[\'K%%d-%s-median\'%%k] for k in range(K)])'%(var,var))
        exec('%s_qp = np.array([QUANTILES[\'K%%d-%s-q+\'%%k] for k in range(K)]) - %s_median'%(var,var,var))
        exec('%s_qm = abs(np.array([QUANTILES[\'K%%d-%s-q-\'%%k] for k in range(K)]) - %s_median)'%(var,var,var))
    Wild_spread_P = {}
    Wild_spread_M = {}
    Short_spread_P = {}
    Short_spread_M = {}
    for var in VARS:
        exec('Wild_spread_M[\'%s\'] = np.where(abs(%s_qm) >= np.median(abs(%s_qm))+abs(%s_qm).std())[0]'%(var,var,var,var))
        exec('Wild_spread_P[\'%s\'] = np.where(abs(%s_qp) >= np.median(abs(%s_qp))+abs(%s_qp).std())[0]'%(var,var,var,var))
        exec('Short_spread_M[\'%s\'] = np.where(abs(%s_qm) <= np.median(abs(%s_qm)))[0]'%(var,var,var))
        exec('Short_spread_P[\'%s\'] = np.where(abs(%s_qp) <= np.median(abs(%s_qp)))[0]'%(var,var,var))
    MEDIAN = {}
    QUANT = {}
    for var in VARS:
        exec('MEDIAN[\'%s\'] = %s_median'%(var,var))
        exec('QUANT[\'%s\'] = zip(%s_qm,%s_qp)'%(var,var,var))
    Q = {}
    for name in ['Wild_spread_P', 'Wild_spread_M', 'Short_spread_P', 'Short_spread_M', 'MEDIAN', 'QUANT']:
        exec('Q[\'%s\'] = %s'%(name,name))
    return Q


def relabel_by_frequence(assign):
    labels, counts = np.unique(assign, return_counts=True)
    new_assign = np.full(len(assign),-1)
    for new_label, label in enumerate(labels[np.argsort(counts)[::-1]]):
        class_slice = np.where(assign == label)[0]
        new_assign[class_slice] = np.full(len(class_slice), new_label)
    return new_assign

def match_classes(labels_1, labels_2):
    labels_1, labels_2 = relabel_by_frequence(labels_1), relabel_by_frequence(labels_2)
    n_clusters = len(np.unique(np.append(labels_1,labels_2)))
    CONT = contingency_matrix(labels_1, labels_2)
    match_1_to_2 = []
    for i_row, row_class in enumerate(CONT):
        SORT_index = [order[0] for order in sorted(enumerate(row_class),key=lambda i:i[1])][::-1]
        control = 0
        i_class = 0
        while((control == 0) & (i_class < len(np.unique(labels_2)))):
            if SORT_index[i_class] not in match_1_to_2:
                match_1_to_2.append(SORT_index[i_class])
                control = 1
            else:
                i_class +=1
            if(i_class > len(np.unique(labels_1))):
                print "No match found to class %d"%i_row
                break
    match_1_to_2 = np.array(match_1_to_2)
    square_matrix = np.zeros((n_clusters,n_clusters))
    CONT = CONT[:,match_1_to_2]
    for i in range(CONT.shape[0]):                   
        square_matrix[i,:CONT.shape[1]] = CONT[i,:]
    Coincidency = np.trace(square_matrix)/float(len(labels_1))
    NORM = np.unique(labels_1, return_counts=True)[1]
    for i_n, norm_value in enumerate(NORM):
        if(norm_value == 0): NORM[i_n] = 1
    norm_contingence_matrix = (square_matrix.transpose()/NORM)
    square_matrix = square_matrix.transpose()
    return match_1_to_2, Coincidency, norm_contingence_matrix, square_matrix


if __name__=='__main__':
    import numpy as np
    x = range(100)
    y = range(100)
    stdpl(x,y)
