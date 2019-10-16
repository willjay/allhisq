import gvar as gv
import corrfitter as cf
import pandas as pd
import numpy as np
import scipy
import pylab as plt
import seaborn as sns

def main():
    print("Analysis functions.")
    
if __name__ == '__main__':
    main()
    
class DataReduction(object):
    """
    A standard data reduction should begin a DataFrame containing
    nominally indepedent samples in each row, with columns
    correpsonding to different correlators. The reduction then has
    three steps:
    * block the data (default is 1 -- no blocking)
    * fold periodic correlators (default is no folding)
    * jackknife (default is 0 elimination -- the full dataset)
    """
    
    def __init__(self, data, names, block_len=1, elim_num=0, fold=None, copy_data=True):
        
        if copy_data:
            self.data  = data.copy()
        else:
            self.data  = data
        self.n_samples = len(data)
        self.names     = names
        self.block_len = block_len
        self.elim_num  = elim_num
        self.maps      = {name : lambda x : list(np.mean(x, axis=0)) for name in names}

        for name in names:
            self.data[name] = self.data[name].apply(np.array)
        
        if fold is not None:
            for name in fold:
                self.data[name] = self.data[name].apply(self.fold)
        
        self.blocked    = self.block()
        self.jackknifed = self.jackknife()
        
        # full dataset, after blocking, with errors
        self.blocked = self.average(self.blocked, jackknife=False)

        # each individual jackknife iteration, with errors
        self.jk_iters = [self.average(jk_iter, jackknife=False) for jk_iter in self.jackknifed]

        # the full reduced dataset (blocking + jackknife), with jackknife errors
        tmp = []
        for jk_iter in self.jk_iters:
            tmp.append({k:v for k,v in gv.mean(jk_iter).iteritems()})
        tmp = pd.DataFrame(tmp)
        self.jk_full = self.average(tmp, jackknife=True)
        
    def block(self):
        """ Blocks the data, returning a single DataFrame. """
        if self.n_samples % self.block_len:
            msg = "Incomensurate n_samples and block_len? Sees ({0},{1})".format(self.n_samples, self.block_len)
            raise ValueError(msg)
        self.n_blocks = self.n_samples / self.block_len    
        # Caution: intentional integer arithmetic
        self.data['group_idx'] = self.data.index // self.block_len
        blocked = self.data.groupby('group_idx').agg(self.maps)
        return blocked

    def jackknife(self):
        """ Carries out jackknife resampling of the data, returning a list of DataFrames. """
        df = self.blocked
        
        if self.elim_num == 0:
            return [df]
        
        if self.n_blocks % self.elim_num:
            msg = "Incomensurate n_blocks and elim_num? Sees ({0},{1})".format(self.n_blocks, self.elim_num)
            raise ValueError(msg)
            
        self.n_drops = self.n_blocks / self.elim_num            
        jk_iters = []
        for drop_idx in np.arange(self.n_drops):
            # Caution: intentional integer arithmetic
            mask = (df.index // self.elim_num != drop_idx)
            jk_iters.append(df[mask])
        
        return jk_iters
        
    def fold(self, arr):
        """ Folds periodic correlator data."""
        try:
            nsamples, nt = arr.shape
            t = np.arange(nt)
            front = arr[:,:nt/2+1]
            back  = arr[:,(nt-t)%nt][:,:nt/2+1]
            new_arr = np.mean([front, back], axis=0)
        except ValueError:
            nt, = arr.shape
            t = np.arange(nt)
            front = arr[:nt/2+1]
            back  = arr[(nt-t)%nt][:nt/2+1]
            new_arr = np.mean([front, back], axis=0)
        return new_arr

    def average(self, df, jackknife=False):
        """
        Averages a DataFrame, returning a correlated dictionary. 
        Args:
            df: the DataFrame
            jackknife: bool, whether or compute the average using the 
                jackknife formula with the "(P+1)" factor.
        """
        d = {}
        for col in df.columns:
            d[col] = gv.mean(np.array(list(df[col].values)))

        # Combines results as independent measurements
        # The errors are the "standard error", i.e., 
        # sigma / Sqrt(n), where sigma is the standard deviation
        correlated = gv.dataset.avg_data(d)

        if jackknife:
            n_samples  = len(df)
            mean       = gv.mean(correlated)
            cov        = gv.evalcov(correlated)
            cov_jk     = {}
            # Two factors are present in the inflation:
            # (1) the critical (P-1) inflation of the covariance matrix, and
            # (2) an additional factor of P, since avg_data() yields
            #     "standard errors" and not standard deviations.
            if hasattr(cov, 'keys'):        
                for key in cov.keys():
                    cov_jk[key] = cov[key]*(n_samples-1)*n_samples
            else:
                 cov_jk = cov*(nsamples-1)
            correlated = gv.gvar(mean, cov_jk)
            
        return correlated

def dumps(var):    
    return '$delim${{{0}}}$delim$'.format(gv.dumps(var, use_json=True)),    
    
def errorbar(ax,x,y,bands=False,**kwargs):
    """
    Wrapper to plot gvars using the matplotlib function errorbar.
    """
    xerr = gv.sdev(x)
    x = gv.mean(x)
    yerr = gv.sdev(y)
    y = gv.mean(y)

    if bands:
        ax.errorbar(x=x,y=y,**kwargs)        
        facecolor=kwargs['color'] if 'color' in kwargs else ax.lines[-1].get_color()
        alpha =kwargs['alpha'] if 'alpha' in kwargs else 1.0
        ax.fill_between(x, y-yerr, y+yerr, facecolor=facecolor, alpha=alpha)
    else:
        ax.errorbar(x=x,xerr=xerr,y=y,yerr=yerr,**kwargs)

def axhline(ax, y, alpha=None, **kwargs):
    if alpha is None:
        alpha = 0.25
    mean = gv.mean(y)
    err  = gv.sdev(y)
    ax.axhline(mean, **kwargs)
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color='k'
    axhspan(ax, y, alpha=alpha, color=color)
    return ax

def axhspan(ax, y, **kwargs):
    mean = gv.mean(y)
    err  = gv.sdev(y)
    ax.axhspan(mean-err, mean+err, **kwargs)
    
def axvline(ax, x, alpha=None, **kwargs):
    if alpha is None:
        alpha = 0.25
    mean = gv.mean(x)
    err  = gv.sdev(x)
    ax.axvline(mean, **kwargs)
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color='k'
    axvspan(ax, x, alpha=alpha, color=color)

def axvspan(ax, x, **kwargs):
    mean = gv.mean(x)
    err  = gv.sdev(x)
    ax.axvspan(mean-err, mean+err, **kwargs)

def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)    
    
def effective_mass(data):
    """        
    Computes the effective mass analytically using the following formula 
    (which I first learned about from Evan Weinberg):

    meff = ArcCosh( (C(t+1)+ C(t-1)) / C(t) )

    This method correctly accounts for contributions both from forward- and
    backward-propagating states. It also work without modification both for
    Cosh-like and Sinh-like correlators.
    """
    return np.arccosh( (data[2:] + data[:-2])/(2.0*data[1:-1]) )

def plot_meff(y, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1)
    m = effective_mass(y)
    x = np.arange(len(m))
    errorbar(ax,x,m,**kwargs)
    return ax
        
def guess_ground_state(y, tmin=None, tmax=None):
    """
    Guess the ground state energy by averaging
    over a "plateau" in the effective mass
    """
    if tmin is None:
        tmin = 0
    if tmax is None:
        tmax = len(y)
    meff    = effective_mass(y[tmin:tmax])    
    plateau = np.mean(meff)
    return gv.mean(plateau)

def round_oom(val):
    """
    Take a number val = m x 10^n and rounds it up to
    the next order of magnitude: 10^(n+1). The number
    n is sometimes called the significand.
    """
    n = int(np.floor(np.log10(abs(val))))
    return 10**(n+1)

def guess_ground_amplitude(y):
    """Guess magnitude of the amplitude of the ground state"""
    return round_oom(gv.mean(y[0]))


def make_prior(n_decay=1, n_oscillate=0, y=None, amps=None, tmin=None, tmax=None, log_dE=False):
    
    if n_decay < 1:
        raise ValueError("Must have n_decay >=1.")
    if n_oscillate < 0:
        raise ValueError("Must have n_oscillate > 0.")
    if amps is None:
        amps = ['a','b','ao','bo']
        
    prior = gv.BufferDict()

    n = range(n_decay)
    if log_dE:
        prior['log(dE)'] = gv.log([gv.gvar('0.5(1.5)') for _ in n])
    else:
        prior['dE'] = [gv.gvar('0.5(1.5)') for _ in n]
    if 'a' in amps:
        prior['a']  = [gv.gvar('1(1)') for _ in n]
    if 'b' in amps:
        prior['b']  = [gv.gvar('1(1)') for _ in n]

    if n_oscillate > 0:
        n = range(n_oscillate)
        if log_dE:
            prior['log(dEo)'] = gv.log([gv.gvar('1.0(1.5)') for _ in n])
        else:
            prior['dEo'] = [gv.gvar('1.0(1.5)') for _ in n]
        if 'ao' in amps:
            prior['ao']  = [gv.gvar('1(1)') for _ in n]
        if 'bo' in amps:
            prior['bo']  = [gv.gvar('1(1)') for _ in n]

    if y is not None:
        ffit = cf.fastfit(y)
        dE_guess  = gv.mean(ffit.E)
        amp_guess = guess_ground_amplitude(y) 
        if log_dE:
            prior['log(dE)'][0] = gv.log(gv.gvar(dE_guess, 0.5*dE_guess))
        else:
            prior['dE'][0] = gv.gvar(dE_guess, 0.5*dE_guess)
        if 'a' in amps:
            prior['a'][0]  = gv.gvar(amp_guess, 1.0*amp_guess)
        elif 'b' in amps:
            prior['b'][0]  = gv.gvar(amp_guess, 1.0*amp_guess)
        else:
            msg = "Error: Unrecognized amplitude structure?"
            raise ValueError(msg)
    
    return prior    
    
def ratio(c3_T, c2_pi, c2_b, nt=None):
    if nt is None: 
        t = c2_pi.tdata
    else:
        t = np.arange(nt)
    T = c3_T.T
    return c3_T.ydata[t] / (c2_pi.ydata[t] * c2_b.ydata[T-t])
    
class ThreePoint(object):
    def __init__(self, nt, ydata, Ts, tdata=None):
        self.ydata = {}
        self.nt = nt
        if tdata is None:
            self.tdata = np.arange(nt)
        for T in Ts:
            if not isinstance(T, int):
                raise TypeError("T must be an integer.")
            self.ydata[T] = ydata[T]
            
    def avg(self, m_hl, m_ll):
        """
        Computes the time-slice-averaged three-point correlation function
        according to Eq. 38 of Bailey et al PRD 79, 054507 (2009) 
        [https://arxiv.org/abs/0811.3640]. This average is useful for 
        suppressing contamination from opposite-parity states.
        Args:
            C3s: array with three-point data and current insertion at fixed time T
            C3_Tp1s: array with three-point data and current insertaion at fixed time (T+1)
            MBs: float, the mass of the "heavy meson"
            Eouts: float, the mass of the "light meson"
            T: integer, the fixed time T associated with the current insertion of C3s
        Returns:
            C3bar: array with the time-slice-averaged correlators
        """
        nt = self.nt 
        c3bar = {}
        for T in self.ydata:
            if T+1 not in self.ydata:
                # Need T and T+1, skip if missing
                continue 
            c3     = self.ydata[T]                   # C(t,   T)
            c3_tp1 = np.roll(c3, -1, axis=0)         # C(t+1, T)
            c3_tp2 = np.roll(c3, -2, axis=0)         # C(t+2, T)
            c3_Tp1     = self.ydata[T+1]             # C(t,   T+1)
            c3_Tp1_tp1 = np.roll(c3_Tp1, -1, axis=0) # C(t+1, T+1)
            c3_Tp1_tp2 = np.roll(c3_Tp1, -2, axis=0) # C(t+2, T+1)
            # Storage for smeared correlator
            tmp = np.empty((nt, ), dtype=gv._gvarcore.GVar)
            for t in range(nt):
                try:
                    tmp[t] =       c3[t] / (np.exp(-m_ll * t) * np.exp(-m_hl * (T - t)))
                    tmp[t] +=      c3_Tp1[t] / (np.exp(-m_ll * t) * np.exp(-m_hl * (T + 1 - t)))
                    tmp[t] += 2. * c3_tp1[t] / (np.exp(-m_ll * (t + 1)) * np.exp(-m_hl * (T - (t + 1))))
                    tmp[t] += 2. * c3_Tp1_tp1[t] / (np.exp(-m_ll * (t + 1)) * np.exp(-m_hl * (T - t)))
                    tmp[t] +=      c3_tp2[t] / (np.exp(-m_ll * (t + 2)) * np.exp(-m_hl * (T - (t + 2))))
                    tmp[t] +=      c3_Tp1_tp2[t] / (np.exp(-m_ll * (t + 2)) * np.exp(-m_hl * (T - t - 1)))
                    tmp[t] *= np.exp(-m_ll * t) * np.exp(-m_hl * (T - t))
                except IndexError:
                    tmp[t] = 0.0
            c3bar[T]  = tmp / 8.

        return c3bar
    
    def __getitem__(self, key):
        return self.ydata[key]

    def __setitem__(self, key, value):
        self.ydata[key] = value
        
    def __len__(self):
        return len(self.ydata)
    
    def __iter__(self):
        for key in self.keys():
            yield key
       
    def iteritems(self):
        return self.ydata.iteritems()
    
    def keys(self):
        return self.ydata.keys()
    
    def values(self):
        return self.ydata.values()
    
class TwoPoint(object):
    
    def __init__(self, tag, ydata, tp, tmin=5, tmax=None, noise_threshy=0.03):
        self.tag = tag
        self.ydata = ydata
        self.tdata = np.arange(len(ydata))
        self.tp = tp
        self.tmin = tmin
        if tmax is None:
            good = gv.sdev(ydata) / gv.mean(ydata) < 0.03
            if not np.all(good):
                tmax = np.argmin(good)
                self.tmax = tmax
            else:
                # Edge case when condition always satisfied.
                self.tmax = len(ydata)
        else:
            self.tmax=tmax
        
        if tp is not None:
            self.nt = np.abs(tp)            
        else:
            self.nt = max(tdata)    
        # Convenient to gather together info regarding times
        self.times = {
            'tp'   : self.tp,
            'tmim' : self.tmin,
            'tmax' : self.tmax,
            'tdata': self.tdata,
        }
        self.ffit = cf.fastfit(G=self.ydata, tp=self.tp, tmin=self.tmin)

    def avg(self):
        """
        Computes the time-slice-averaged two-point correlation function
        according to Eq. 37 of Bailey et al PRD 79, 054507 (2009) 
        [https://arxiv.org/abs/0811.3640]. This average is useful for 
        suppressing contamination from opposite-parity states.
        Args:
            C2s: array with two-point correlator data
            MOs: the ground-state mass of the correlator
        Returns
            C2bar: array with the time-slice-averaged correlators
        """

        c2 = self.ydata
        tmax = len(c2) 
        m  = gv.mean(self.ffit.E)
        c2_tp1s = np.roll(self.ydata, -1, axis=0)
        c2_tp2s = np.roll(self.ydata, -2, axis=0)
        c2bar = np.empty((tmax,), dtype=gv._gvarcore.GVar)
        for t in range(tmax):
            c2bar[t] = c2[t] / np.exp(-m * t)
            c2bar[t] += 2 * c2_tp1s[t] / np.exp(-m * (t + 1))
            c2bar[t] += c2_tp2s[t] / np.exp(-m * (t + 2))
            c2bar[t] *= np.exp(-m * t)
        return c2bar / 4.        
                
    def __getitem__(self, key):
        return self.ydata[key]

    def __setitem__(self, key, value):
        self.ydata[key] = value
        
    def __len__(self):
        return len(self.ydata)

class FormFactorDataset(object):
    def __init__(self, ds, ns, nt, times=None):

        if times is None:
            tmin_ll = 5
            tmin_hl = 5
            tmax_ll = None
            tmax_hl = None
        else:
            tmin_ll = times['light-light:tmin']
            tmin_hl = times['heavy-light:tmin']
            tmax_ll = times['light-light:tmax']
            tmax_hl = times['heavy-light:tmax']
        
        self.ns = ns
        self.nt = nt
        tp = nt # Not true in general, but always true for these analyses?
        
        # Structure the data with objects
        self.c2 = {
            'light-light' : TwoPoint(tag='light-light', ydata=ds['light-light'], tp=tp, tmin=tmin_ll, tmax=tmax_ll),
            'heavy-light' : TwoPoint(tag='heavy-light', ydata=ds['heavy-light'], tp=tp, tmin=tmin_hl, tmax=tmax_hl),
        }
        
        # Convenient references
        self.c2_ll = self.c2['light-light']
        self.c2_hl = self.c2['heavy-light']
        
        self.Ts = sorted([T for T in ds.keys() if isinstance(T, int)])
        self.c3 = ThreePoint(nt=self.nt, Ts=self.Ts, ydata=ds)
        
        # Compute smeared quantities
        self.analyze2pt()
        self.analyze3pt()
        self.analyze_ratio()
    
    def keys(self):
        return self.c2.keys() + self.c3.keys()
    
    def values(self):
        return [self[key] for key in self.keys()]
    
    def __getitem__(self, key):
        if key in self.c2.keys():
            return self.c2[key]
        elif key in self.c3.keys():
            return self.c3[key]
        else:
            msg = "Unrecognized key in FormFactorData: {0}".format(key)
            raise KeyError(msg)
    
    def __iter__(self):
        for key in self.keys():
            yield key

    def analyze2pt(self):
        self.m = {} 
        self.m['light-light'] = gv.mean(self.c2['light-light'].ffit.E)
        self.m['heavy-light'] = gv.mean(self.c2['heavy-light'].ffit.E)

        self.c2bar = {}
        self.c2bar['light-light'] = self.c2['light-light'].avg()
        self.c2bar['heavy-light'] = self.c2['heavy-light'].avg()

        # Convenient references
        self.m_ll = self.m['light-light']
        self.m_hl = self.m['heavy-light']
        self.c2bar_ll = self.c2bar['light-light']
        self.c2bar_hl = self.c2bar['heavy-light']
        
    def analyze3pt(self):
        self.c3bar = self.c3.avg(m_ll=self.m_ll, m_hl=self.m_hl)
            
    def analyze_ratio(self):
        self.rbar = self.avg_ratio()
        self.r    = self.ratio()

    def avg_ratio(self):
        """
        Computes the time-slice-averaged ratio of correlation functions
        according to Eq. 39 of Bailey et al PRD 79, 054507 (2009) 
        [https://arxiv.org/abs/0811.3640]. This average is useful for 
        suppressing contamination from opposite-parity states.
        """
        m_ll = self.m_ll
        m_hl = self.m_hl
        c2bar_ll = self.c2bar_ll
        c2bar_hl = self.c2bar_hl
        c3bar = self.c3bar
        
        # Get shortest necessary time
        ts = [self.c2_ll.tdata, self.c2_hl.tdata]
        t = ts[np.argmin([len(ti) for ti in ts])]
        tmax = max(t)
        rbar = {}
        
        for T, c3bar_T in c3bar.iteritems():
            denom = np.sqrt(c2bar_ll[t]*c2bar_hl[T-t]*np.exp(-m_ll*t)*np.exp(-m_hl*(T-t))) 
            tmax = min(len(c3bar_T), tmax)
            rbar[T] = c3bar_T[:tmax] * np.sqrt(2*m_ll) / denom[:tmax]

        return rbar
    
    def ratio(self): #c3_T, c2_pi, c2_b, nt=None):
            
        # Get shortest necessary time
        ts = [self.c2_ll.tdata, self.c2_hl.tdata]
        t = ts[np.argmin([len(ti) for ti in ts])]
        tmax = max(t)
        r = {}
        for T, c3_T in self.c3.iteritems():
            denom = np.sqrt(self.c2_ll[t] * self.c2_hl[T-t] *\
                            np.exp(-self.m_ll *t) * np.exp(-self.m_hl * (T-t)))
            r[T] = c3_T[t] * np.sqrt(2.0 * self.m_ll) / denom
            
        return r                
                        
    def plot(self, ax, y, **kwargs):
        x = np.arange(len(y))
        errorbar(ax,x,y, **kwargs)           
            
    def plot_corr(self, ax=None):    
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(10,5))
        colors = sns.color_palette()
        # pi
        self.plot(ax, self.c2_ll[:], fmt='.',color=colors[0])
        self.plot(ax, self.c2bar_ll[:], color=colors[0])

        # b
        self.plot(ax, self.c2_hl[:], fmt='.', color=colors[1])
        self.plot(ax, self.c2bar_hl[:], color=colors[1])

        # 3pt
        for count, T in enumerate(self.c3):
            self.plot(ax, self.c3[T][:], fmt='.', color=colors[2+count])
            if T in self.c3bar:         
                self.plot(ax, self.c3bar[T][:], color=colors[2+count])
            
        ax.set_yscale('log')            
        ax.set_xlabel('t')
        ax.set_ylabel('C(t)')
        return ax
    
    def plot_ratio(self, ax=None, xmin=0, xmax=None, bands=False, **plot_kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(10,5))
        colors = sns.color_palette()
        if xmax is None:
            xmax = max(self.Ts)
        x = np.arange(xmin, xmax)
        for color, T in zip(colors, sorted(self.r)):
            # Unsmeared "saw-tooth" ratio
            label = "R, T={0}".format(T)     
            errorbar(ax, x, self.r[T][xmin:xmax], label=label, color=color, fmt='-.', **plot_kwargs) 

            # Smeared ratio
            if T in self.rbar:
                label = "Rbar, T={0}".format(T)           
                errorbar(ax, x, self.rbar[T][xmin:xmax], label=label, color=color, bands=bands, **plot_kwargs)

        ax.set_xlabel('t/a')
        ax.set_ylabel('$\\bar{R}$ (lattice units)')
        ax.legend(loc=0)
        return ax

class FormFactorPrior(object):
    """
    Let's keep it focused. We don't need a class which builds a prior for 
    every possible analysis. We need one for *this* analysis. The names of
    the parameters will always be the same. What changes is the data
    
    nstates : dict, {'light-light' : (3,2) 'heavy-light' : (3,2)}
    
    """
    def __init__(self, ds, nstates, widths=None):
    
        self.c2 = ds.c2
        self.c3 = ds.c3
        self.rbar = ds.rbar
        self.nstates = nstates
        if widths is None:
            self.widths = {}
        else:
            self.widths = widths
        #TODO check nstates for valid setup

        for (n_decay, n_oscillate) in nstates.values():
            if n_decay < 1:
                raise ValueError("Must have n_decay >=1.")
            if n_oscillate < 0:
                raise ValueError("Must have n_oscillate > 0.")

        self.prior = {}
        for tag, (n_decay, n_oscillate) in nstates.iteritems():
            self.make_2pt_prior(tag, n_decay, n_oscillate)

        self.make_3pt_prior()
        
    def __getitem__(self, key):
        return self.prior.__getitem__(key) 
                
    def __setitem__(self, key, value):
        self.prior.__setitem__(key, value)
        
    def __len__(self):
        return self.prior.__len__()
    
    def __iter__(self):
        for key in self.keys():
            yield key

    def __str__(self):
        return self.prior.__str__()

    def iteritems(self):
        return self.prior.iteritems()
    
    def keys(self):
        return self.prior.keys()
    
    def values(self):
        return self.prior.values()
    
    def get_gaussian(self, key):
        """ Get the gaussian version of the key """
        if key.startswith('log'):
            raise ValueError("Use get_lognormal(key) for log-normal parameters.")
        if key in self.prior:
            return self.prior[key]
        else:
            return np.exp(self.prior[self.alias(key)])        
        
    def get_lognormal(self, key):
        """ Get the lognormal version of the key """
        if not key.startswith('log'):
            raise ValueError("Use get_gaussian(key) for gaussian parameters.")
        if key in self.prior:
            return self.prior[key]
        else:
            return np.log(self.prior[self.alias(key)])
        
    def make_2pt_prior(self, tag, n_decay, n_oscillate):
        
        default_widths = {
            'dE' : 0.5,
            'a'  : 1,
            'dEo': 0.2,
            'ao' : 0.2,
        }
        
        # Overriding defaults
        for key in default_widths:
            if key in self.widths:
                default_widths[key] = self.widths[key]

        defaults = {
            'dE' : '0.5({0})'.format(default_widths['dE']),
            'a'  : '1({0})'.format(default_widths['a']),
            'dEo': '0.5({0})'.format(default_widths['dEo']),
            'ao' : '0.2({0})'.format(default_widths['ao']),
        }
              
        tmp_prior = {}
        # General decaying states
        n = range(n_decay)
#         tmp_prior['dE'] = [gv.gvar('0.5(0.5)') for _ in n]
        tmp_prior['dE'] = [gv.gvar('0.5(0.5)')] + [gv.gvar(defaults['dE']) for _ in n[1:]]
#         tmp_prior['a']  = [gv.gvar('1(1)') for _ in n]
        tmp_prior['a']  = [gv.gvar(defaults['a']) for _ in n]        
#         tmp_prior['a']  = [gv.gvar('0.25(0.25)')] + [gv.gvar('0.2(0.2)') for _ in n[1:]]
        # General oscillating states
        if n_oscillate > 0:
            n = range(n_oscillate)
#             tmp_prior['dEo'] = [gv.gvar('1.0(1.0)') for _ in n]
            tmp_prior['dEo'] = [gv.gvar('1.0(1.0)')] + [gv.gvar(defaults['dEo']) for _ in n[1:]]                        
#             tmp_prior['dEo'] = [gv.gvar('0.5(0.5)')] + [gv.gvar('0.5(0.2)') for _ in n[1:]]            
#             tmp_prior['ao']  = [gv.gvar('1(1)') for _ in n]
            tmp_prior['ao']  = [gv.gvar('0.25(0.25)')] + [gv.gvar(defaults['ao']) for _ in n[1:]]

        # Guesses from data for ground-state energy and amplitude
        dE_guess  = gv.mean(self.c2[tag].ffit.E)
        amp_guess = gv.mean(self.c2[tag].ffit.ampl)
        dE_err  = gv.sdev(tmp_prior['dE'][0])
        amp_err = gv.sdev(tmp_prior['a'][0])
        
        tmp_prior['dE'][0] = gv.gvar(dE_guess,  dE_err)
        tmp_prior['a'][0]  = gv.gvar(amp_guess, amp_err)

        # Rename and add to final prior
        for key in ['a','ao','dE','dEo']:
            new_key = "{tag}:{key}".format(tag=tag,key=key)
            self.prior[new_key] = tmp_prior[key]

    def make_3pt_prior(self):
            
        n, no = self.nstates['light-light']
        m, mo = self.nstates['heavy-light']

        # General guesses
        tmp_prior = {}
        tmp_prior['Vnn'] = gv.gvar(n * [m * ['0.1(2.0)']])
        tmp_prior['Vno'] = gv.gvar(n * [mo* ['0.1(2.0)']])
        tmp_prior['Von'] = gv.gvar(no * [m * ['0.1(2.0)']])
        tmp_prior['Voo'] = gv.gvar(no * [mo* ['0.1(2.0)']])

        maxes = []
        for T,val in self.rbar.iteritems():
            local_max = max(gv.mean(val[:T-2]))
            maxes.append(local_max)
        r_guess = max(maxes)
        r_width = np.abs(max(maxes) - min(maxes))
        
        # Average across "plateaus" in Rbar
#         r_plateau = {}
#         for T in self.rbar:
#             tmin = self.c2['light-light'].tmin
#             tmax = T - self.c2['heavy-light'].tmin
#             r_plateau[T] = np.mean(self.rbar[T][tmin:tmax])
#         r_guess = np.mean(gv.mean(r_plateau.values()))
#         self.r_guess = r_guess

        # Convert to V
        m_ll    = gv.mean(self.c2['light-light'].ffit.E)
        v_guess = r_guess / np.sqrt(2.0*m_ll)
        v_width = r_width / np.sqrt(2.0*m_ll)
        tmp_prior['Vnn'][0,0] = gv.gvar(v_guess, v_width)

        for key, val in tmp_prior.iteritems():
            self.prior[key] = val

    def update(self, update_with, width=0.1):
        """
        Updates the central values of the existing prior 
        """            
        def handle_lognormal(logged_value, width):
            value = gv.exp(logged_value) 
            return gv.log(handle_normal(value, width))
        
        def handle_normal(value, width):
            mean = gv.mean(value)
            sdev = np.maximum(gv.sdev(value), width)
            return gv.gvar(mean, sdev)
                
        for key in update_with:
            if key in self.prior:    
                if key.startswith('log('):
                    self.prior[key] = handle_lognormal(update_with[key], width)
                else:
                    self.prior[key] = handle_normal(update_with[key], width)
                
    def positive_params(self):
        """
        Switch to log-normal priors to ensure positive parameters        
        """
        for key in self.prior:
            # Not necessary to touch anything of the form 'log(...)'
            if not key.startswith('log'):  
                # Switch energies and amplitudes to logs
                if (':dE' in key) or (':a' in key):
                    new_key = 'log({key})'.format(key=key)
                    self.prior[new_key] = gv.log(self.prior.pop(key))

    def positive_params_external(self, external):
        for key in external:
            # Not necessary to touch anything of the form 'log(...)'
            if not key.startswith('log'):  
                # Switch energies and amplitudes to logs
                if (':dE' in key) or (':a' in key):
                    new_key = 'log({key})'.format(key=key)
                    external[new_key] = gv.log(external.pop(key))
        return external
    
    def alias(self, pname):
        if pname.startswith('log'):
            return pname.replace('log','').lstrip('(').rstrip(')')
        else:
            return 'log({0})'.format(pname)
                    
    def p0(self):
        return {k : gv.mean(v) for k,v in self.prior.iteritems()}
    
    
class FormFactorAnalysis(object):
    
    def __init__(self, data_reduced, ns, nt, times, nstates, pdg_prior=None, do_jackknife=False, **fitter_kwargs):
        
        if do_jackknife:
            raise NotImplemented("Jackknife not yet implemented.")
        
        self.ds = FormFactorDataset(data_reduced.blocked, ns, nt, times=times)
        
#         self.data = data
#         self.ns = ns
#         self.nt = nt
#         ds = gv.dataset.avg_data(data)

        # FormFactorDataset
#         self.ds = FormFactorDataset(ds, ns, nt, times=times)
        self.nstates = nstates
        self.prior = FormFactorPrior(self.ds, nstates)
        self.fits = {}
        self.fitter = None # Set below by three-point fit

        if pdg_prior is not None:
            pdg_prior = self.prior.positive_params_external(pdg_prior)
            self.prior.update(update_with=pdg_prior)
        
        # Two-point fits
        for tag in self.ds.c2:
            fit = self.fit_two_point(tag, **fitter_kwargs)
            if fit is None:
                print('[-] Warning: {0} fit failed'.format(tag))
            else:
                self.prior.update(update_with=fit.p)                
            self.fits[tag] = fit

        # The full joint three-point
        fit = self.fit_three_point(**fitter_kwargs)
        if fit is None:
            print('[-] Warning: full fit failed')
        self.fits['full'] = fit
            
        # Jackknife analysis
        if do_jackknife:
            self.jk_fits = self.fit_jackknife(n_elim=2**5*3, **fitter_kwargs)  
        else:
            self.jk_fits = None
           
        # Collect diagnostics
        self.stats = {}
        for tag, fit in self.fits.iteritems():
            if fit is not None:
                self.stats[tag] = FitStats(fit)
            else:
                self.stats[tag] = None
         
        # Convert results about V[0,0] into the ratio "Rbar"
        # which is supposed to deliver the form factor 
        m_pi    = self.prior.get_gaussian('light-light:dE')[0]
        v_guess = self.prior['Vnn'][0,0] 
        self.r_guess = v_guess*np.sqrt(2.0*m_pi)

        m_pi = self.fits['full'].p['light-light:dE'][0]
        v    = self.fits['full'].p['Vnn'][0,0]
        self.r =  v*np.sqrt(2.0*m_pi)
            
    def fit_two_point(self, tag, **fitter_kwargs):
    
        model = self.get_model(tag)
        prior = self.prior
        prior.positive_params() 
        p0 = prior.p0()
        fitter = cf.CorrFitter(models=model)
        fit = fitter.lsqfit(data=self.ds, prior=prior, p0=p0, **fitter_kwargs)
        if np.isnan(fit.chi2):
            print('[+] {0} failed once. Trying once more with updated prior.'.format(tag))
            # Try improving priors on oscillating states and re-running
            _, n_oscillating = self.nstates['{0}'.format(tag)]
            for idx in np.arange(n_oscillating):
                prior['log({0}:dEo)'.format(tag)][idx] = gv.gvar(prior['log({0}:dE)'.format(tag)][idx])
                p0 = prior.p0()
                fit = fitter.lsqfit(data=self.ds, prior=prior, p0=p0, **fitter_kwargs)
        if np.isnan(fit.chi2):
            fit = None
        return fit

    def fit_three_point(self, **fitter_kwargs):        
        models = [self.get_model(tag) for tag in self.ds]
        models = [model for model in models if model is not None]
        if len(models) >= 3:
            prior  = self.prior
            prior.positive_params()
            fitter = cf.CorrFitter(models=models)
            self.fitter = fitter
            fit = fitter.lsqfit(data=self.ds, prior=prior, **fitter_kwargs)
        else:
            fit = None

        return fit        
      
    def fit_jackknife(self, n_elim=1, **fitter_kwargs):
        jk = Jackknife(self.data, n_elim)
        jk_fits = []
        for jk_ds in jk:
            jk_fits.append(self.fitter.lsqfit(data=jk_ds, prior=self.prior, **fitter_kwargs))
        return jk_fits
    
    def get_model(self, tag):
        
        if isinstance(self.ds[tag], TwoPoint):

            a_pnames  = ('{0}:a'.format(tag), '{0}:ao'.format(tag))
            b_pnames  = ('{0}:a'.format(tag), '{0}:ao'.format(tag))
            dE_pnames = ('{0}:dE'.format(tag),'{0}:dEo'.format(tag))
    
            two_point = self.ds[tag]
            model = cf.Corr2(
                datatag = two_point.tag,
                tp      = two_point.tp,
                tmin    = two_point.tmin,
                tmax    = two_point.tmax,
                tdata   = two_point.tdata,
                a  = a_pnames,
                b  = b_pnames,
                dE = dE_pnames,
                s  = (1.0, -1.0)
            )
            return model
        elif isinstance(tag, int):
            three_point = self.ds[tag]
            T = tag
            tfit = np.arange(self.ds['heavy-light'].tmin, T - self.ds['light-light'].tmin)
            if len(tfit) > 0:
                tdata = self.ds.c3.tdata
                model = cf.Corr3(
                    datatag=T,
                    T     = T,
                    tdata = tdata,
                    tfit  = tfit,
                    # Amplitudes in light-light 2-pt function
                    a=('light-light:a',   'light-light:ao'), 
                    # Amplitudes in heavy-light 2-pt function
                    b=('heavy-light:a',   'heavy-light:ao'), 
                    # Energies in light-light 2-pt function
                    dEa=('light-light:dE','light-light:dEo'),
                    # Energies in light-light 2-pt function
                    dEb=('heavy-light:dE','heavy-light:dEo'),
                    # sign factors in light-light 2-pt function
                    sa=(1.0,-1.0), 
                    # sign factors in heavy-light 2-pt function
                    sb=(1.0,-1.0), 
                    # connect light-light decay --> heavy-light decay
                    Vnn='Vnn', 
                    # connect light-light decay --> heavy-light oscillating
                    Vno='Vno', 
                    # connect light-light oscillating --> heavy-light decay
                    Von='Von', 
                    # connect light-light oscillating --> heavy-light oscillating
                    Voo='Voo'  
                )
            else:
                model = None
            return model
        else:
            raise TypeError("get_model() needs TwoPoint or ThreePoint objects.")

            
    def plot_results(self, axarr=None):

        nrows = len(self.fitter.models)
        if axarr is None:
            fig, axarr = plt.subplots(nrows=nrows, sharex=True, figsize=(10,10))
        elif len(axarr) < nrows:
            raise ValueError("Too few rows for plot_results()?")
 
        fit = self.fits['full']

        for ax, model in zip(axarr, self.fitter.models):
            tag = model.datatag
            tfit = model.tfit    
            ratio = self.ds[tag][tfit] / fit.fcn(fit.p)[tag]
            errorbar(ax, x=tfit, y=ratio, fmt='.')
            ax.axhline(1.0, ls='--', color='k')
            ax.set_ylabel(tag)
            ax.set_title('data/fit')
        
        axarr[-1].set_xlabel('t/a')
        return fig, axarr

    def plot_energy(self, ax, tag, do_partners=False):
        """ Make summary plot of the energies """

        if not do_partners: 
            energy_tag = '{0}:dE'.format(tag)
            title = "Energies: {tag}".format(tag=tag)
            
            # Effective mass
            plot_meff(self.ds[tag], ax=ax, fmt='.', label='Effective mass')
            plot_meff(self.ds.c2bar[tag][:-5], ax=ax, fmt='.', color='k', label='Smeared effective mass')

            # Fastfit guess
            E_ffit = self.ds[tag].ffit.E
            axhline(ax, E_ffit, label='ffit guess', color='k', ls=':')

        else:
            energy_tag = '{0}:dEo'.format(tag)
            title = "Partner Energies: {tag}".format(tag=tag)

        # Fit masses
        Es     = np.cumsum(self.fits['full'].p[energy_tag])
        colors = sns.color_palette(n_colors=len(Es))
        for idx, E in enumerate(Es):
            label = "Fit: E{0}".format(idx)
            axhline(ax, E, label=label, alpha=0.75, color=colors[idx])
    
        # Priors
        Es = np.cumsum(self.prior.get_gaussian(energy_tag))
        for idx, E in enumerate(Es):
            label = "Prior: E{0}".format(idx)
            axhspan(ax, E, label=label, alpha=0.25, color=colors[idx])
    
        # Formatting
        ax.set_title(title)
        ax.set_xlabel("$t/a$")
        ax.set_ylabel("$Ea$")
        ax.legend(loc=1)
        return ax

    def plot_amplitude(self, ax, tag, do_partners=False):
        """ Make summary plot of the amplitudes"""

        if not do_partners: 
            amp_tag = '{0}:a'.format(tag)
            title = "Amplitudes: {tag}".format(tag=tag)

            # Effective amplitude A_eff = C(t)*Exp(m_eff*t)
            corr = self.ds[tag]
            t    = corr.tdata
            meff = effective_mass(corr)
            x = t[1:-1]
            y = np.sqrt(np.exp(meff*x) * corr[1:-1])
            errorbar(ax, x, y, fmt='.', label='Effective amplitude')

            # Fastfit guess
            amp_ffit = np.sqrt(corr.ffit.ampl)
            axhline(ax, amp_ffit, label='ffit guess', color='k', ls=':')    
        
        else:
            amp_tag = '{0}:ao'.format(tag)
            title = "Partner Amplitudes: {tag}".format(tag=tag)

        # Fit amplitudes
        amps   = self.fits['full'].p[amp_tag]
        colors = sns.color_palette(n_colors=len(amps))
        for idx, amp in enumerate(amps):
            label = 'Fit: A{0}'.format(idx)
            axhline(ax, amp, label=label, color=colors[idx])
        
        # Priors
        amps = self.prior.get_gaussian(amp_tag)
        for idx, amp in enumerate(amps):
            label = "Prior: A{0}".format(idx)
            axhspan(ax, amp, label=label, alpha=0.25, color=colors[idx])
        
        if not do_partners:
            # Effective amplitude unstable at long times
            # Set limits by hand as precautionary measure
            ax.set_ylim(ymin=0.0, ymax=1.0) 
            
        # Formatting
        ax.set_title(title)
        ax.set_xlabel("$t/a$")
        ax.set_ylabel("Amplitude (lattice units)")
        ax.legend(loc=1)
        return ax
    
    def plot_states(self, axarr=None, do_partners=False):

        if axarr is None:
            fig, axarr = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(20,20))
        ((ax1,ax2),(ax3,ax4)) = axarr

        # Masses in first row
        for ax, tag in zip([ax1,ax2],['light-light', 'heavy-light']):
            _ = self.plot_energy(ax, tag, do_partners)

        # Amplitudes in second row
        for ax, tag in zip([ax3,ax4],['light-light', 'heavy-light']):            
            _ = self.plot_amplitude(ax, tag, do_partners)

        # Bands for fit range    
        ax_cols = [(ax1,ax3), (ax2, ax4)]
        for tag, ax_col in zip(['light-light','heavy-light'], ax_cols):
            for ax in ax_col:
                tmin = self.ds[tag].tmin
                tmax = self.ds[tag].tmax
                axvline(ax, tmin, color='k', ls='--')
                axvline(ax, tmax, color='k', ls='--')    

        fig.tight_layout()

        return axarr            
        
    def plot_form_factor(self, ax=None, xmax=12, color=None):
        """ 
        Plot the ratio which delivers the form factor together
        with the prior estimate and fit result.
        """
        if color is None:
            color = 'k'
        
        ax = self.ds.plot_ratio(ax=ax, xmax=xmax)
        axhspan(ax, self.r_guess, alpha=0.25, color=color, label='Prior: R')
        axhline(ax, self.r,       alpha=0.50, color=color, label='Fit: R')
        ax.set_title("Form factor compared with estimates")  
        ax.legend(loc=1)   
        return ax
        
class FitStats(object):
    def __init__(self, fit):
        
        self.chi2     = self.compute_chi2(fit, aug=False)
        self.chi2_aug = self.compute_chi2(fit, aug=True)
        self.nparams  = self.count_nparams(fit.p)
        self.ndata    = self.count_ndata(fit.y)
        self.Q        = self.correlated_Q(self.chi2_aug, self.ndata)
        self.p_value  = self.correlated_p(self.chi2,     self.ndata, self.nparams)

    def count_nparams(self, params):
        """
        Counts the number of fit parameters np, being careful
        to avoid double counting of "log priors" are present.
        """
        nparams = 0
        for pname, val in params.iteritems():
            log_pname = 'log({0})'.format(pname)
            if log_pname in params.keys():
                # Skip this parameter
                continue
            if hasattr(val, '__len__'):
                nparams += len(np.asarray(val).flatten())
            else:
                nparams += 1
        return nparams

    def count_ndata(self, ydata):
        """ Counts the number of data points nd """
        ndata = 0
        if hasattr(ydata, 'keys'):
            for key in ydata.keys():
                ndata += len(np.asarray(ydata[key]).flatten())
        else:
            ndata = len(np.asarray(ydata).flatten()) 
        return ndata

    def correlated_Q(self, chi2_aug, ndata):
        """
        Computes the correlated Q-value using the survival function (sf), 
        i.e. the complementary cumulative distribution function (CCDF).
        See Appendix B of A. Bazavov et al., PRD 93, 113016 (2016).
        """
        return scipy.stats.distributions.chi2.sf(chi2_aug, ndata)

    def correlated_p(self, chi2, ndata, nparams):
        """
        Computes the correlated p-value using the survival function (sf), 
        i.e. the complementary cumulative distribution function (CCDF).
        See Appendix B of A. Bazavov et al., PRD 93, 113016 (2016).    
        """
        nu = ndata-nparams
        return scipy.stats.distributions.chi2.sf(chi2, nu)

    def compute_chi2(self, fit, aug=False, trust_lsqfit=False):

        def _chi2(yfit, ydata):
            # Get the fit values, data, and covariance matrix as dictionaries        
            cov_dict = gv.evalcov(ydata)

            # Enforce an ordering of keys
            klist = sorted(ydata.keys())

            # Reserve space for arrays
            # Implementation note: flatten allows for the special case
            # of matrix-valued priors, e.g., for the transition matrix Vnn
            sizes = [len(ydata[key].flatten()) for key in klist]
            nd    = sum(sizes)
            diff  = np.empty(nd)
            cov   = np.zeros((nd, nd))

            # Get start and end points for intervals
            ends = np.cumsum(sizes)
            starts = ends - sizes

            # Populate arrays
            for start_i, end_i, key_i in zip(starts, ends, klist):
                diff[start_i:end_i] = gv.mean(ydata[key_i] - yfit[key_i]).flatten()
                for start_j, end_j, key_j in zip(starts, ends, klist):
                    try:
                        cov[start_i:end_i, start_j:end_j] = cov_dict[(key_i, key_j)]
                    except ValueError: 
                        # Implementation note: matrix-valued priors have multi-dimensional
                        # covariance matrices, which must be reshaped in a 2x2 array
                        cov[start_i:end_i, start_j:end_j] = cov_dict[(key_i, key_j)].\
                                                            reshape(end_i-start_i,end_j-start_j)

            # The "usual" chi2 function (ydata-yfit).cov_inv.(ydata-yfit)
            try:
                chi2 = np.dot(diff, np.linalg.solve(cov, diff))
            except np.linalg.LinAlgError:
                chi2 = np.nan
            return chi2

        if aug and trust_lsqfit:
            # lsqfit returns the augmented chi2
            return fit.chi2

        # Standard chi2, without the term involving the prior
        chi2 = _chi2(fit.fcn(fit.p), fit.y)

        if aug:
            # Augmeted chi2, including the term with the prior
            chi2 += _chi2(fit.p, fit.prior)

        return chi2    
            
class Jackknife(object):
    def __init__(self, data, n_elim):
        
        if hasattr(data, 'values'):
            n_configs = np.unique([len(val) for val in data.values()]).item()
        else: 
            n_configs = len(data)
        if n_configs % n_elim:
            msg = "Incomensurate nconfigs and n_elim? Sees ({0},{1})".format(n_configs, n_elim)
            raise ValueError(msg)

        self.data      = data
        self.n_elim    = n_elim
        self.n_configs = n_configs
        self.n_drops   = n_configs / n_elim    
        self.counts    = np.arange(n_configs)

    def mask_data(self, data, mask):
        if hasattr(data, 'keys'):
            return gv.dataset.avg_data({k : v[mask] for k,v in data.iteritems()})        
        else:
            return gv.dataset.avg_data(data[mask])
    def __iter__(self):
        
        for drop_idx in np.arange(self.n_drops):
            # Caution: intentional integer arithmetic
            mask  = (self.counts/self.n_elim != drop_idx) 
            yield self.mask_data(self.data, mask)
            
            
            
class PDGSpectrum(object):
    
    def __init__(self, params, a):

        self.params = params
        self.a = a
        self.load_data() # initialize self.mesons, self.quarks
        self.mesons['ma'] = self.mesons['m (MeV)'].\
            apply(lambda m : self.to_lattice_units(m,a))
        self.quarks['ma'] = self.quarks['m (MeV)'].\
            apply(lambda m : self.to_lattice_units(m,a))
    
        # Identify simulated quarks within the physical spectrum    
        quarks_simulated = []
        for tag in ['heavy_mass','spectator_quark_mass','antiquark_mass']:
            ma_simulated = params[tag]
            name_physical, ma_physical = self.get_closest_quark(ma_simulated)
            quarks_simulated.append([tag, name_physical, ma_physical, ma_simulated])
        self.quarks_simulated = pd.DataFrame(
            quarks_simulated,
            columns=['name_simulated','name_physical','ma_physical','ma_simulated']
        )
        
    def load_data(self):
        self.mesons = pd.DataFrame([
            # Light mesons
            ['pi0',       134.997,'1/2(0-)', 'll'],
            ['eta',       547.862,'0+(0-+)', 'll'],
            ['f0(500)',   450.,   '0+(0++)', 'll'],
            ['rho(770)',  775.26, '1+(1--)', 'll'],
            ['omega(782)',782.65, '0-(1--)', 'll'],
            ['etaprime(958)',957.78,'0+(0-+)', 'll'],
            ['f0(980)',   990.,   '0+(0++)', 'll'],
            ['phi(1020)', 1019.46,'0-(1--)', 'll'],
            ['h1(1170)',  1170.,  '0-(1+-)', 'll'],
            ['b1(1235)',  1229.5, '1+(1+-)', 'll'],
            ['a1(1260)',  1230.,  '1-(1++)', 'll'],
            ['f2(1270)',  1275.5, '0+(2++)', 'll'],
            ['f1(1285)',  1281.9, '0+(1++)', 'll'],
            ['eta(1295)', 1294.,  '0+(0-+)', 'll'],
            ['pi(1300)',  1300.,  '1-(0-+)', 'll'],
            ['a2(1320)',  1318.3, '1-(2++)', 'll'],
            ['eta(1405)', 1408.8, '0+(0-+)', 'll'],
            # Strange Mesons
            ['K(+/-)',    493.677,'1/2(0-)', 'sl'],
            ['K0',        497.611,'1/2(0-)', 'sl'],
            ['K0*(700)',  730.,   '1/2(0+)', 'sl'],
            ['K*(892)',   891.76, '1/2(1-)', 'sl'],
            ['K1(1270)',  1270,   '1/2(1+)', 'sl'],
            ['K1(1400)',  1403,   '1/2(1+)', 'sl'],
            ['K*(1410)',  1421,   '1/2(1-)', 'sl'],
            ['K0*(1430)', 1425,   '1/2(0+)', 'sl'],
            ['K2*(1430)', 1425,   '1/2(2+)', 'sl'],
            # Charmed mesons
            ['D',         1864.83,'1/2(0-)', 'cl'],
            ['D*(2007)',  2006.85,'1/2(1-)', 'cl'],
            ['D*(2400)',  2318,   '1/2(0+)', 'cl'],
            ['D1(2420)',  2420.8, '1/2(1+)', 'cl'],
            ['D1(2430)',  2427.,  '1/2(1+)', 'cl'],
            ['D2*(2460)', 2460.,  '1/2(2+)', 'cl'],
            ['D(2550)',   2564,   '1/2(??)', 'cl'],
            # Charmed, Strange Mesons
            ['Ds(+/-)',   1968.34,'0(0-)', 'cs'],
            ['Ds(*)',     2112.2, '0(??)', 'cs'],
            ['Ds0(2317)', 2317.7, '0(0+)', 'cs'],
            ['Ds1(2460)', 2459.5, '0(1+)', 'cs'],
            ['Ds1(2536)', 2535.1, '0(1+)', 'cs'],
            ['Ds2*(2573)',2569.1, '0(2+)', 'cs'],
            ['Ds1*(2700)',2708.3, '0(1-)', 'cs'],
            #Bottom, Strange Mesons
            ['Bs(0)',     5366.89,'0(0-)', 'bs'],
            ['Bs(*)',     5415.3, '0(1-)', 'bs'],
            ['Bs1(5830)', 5828.63,'0(1+)', 'bs'],
            ['Bs2(5840)', 5848,   '0(2+)', 'bs']],
            columns=['name','m (MeV)', 'quantum_numbers', 'quark_content']
        )
        
        for tag in ['I','G','J','P','C']:
            self.mesons[tag] = self.mesons['quantum_numbers'].\
                                apply(self.parse_quantum_numbers).\
                                apply(lambda d: d[tag])

        self.quarks = pd.DataFrame([
            ['u', 2.2],
            ['d', 4.7],
            ['l', 3.5],
            ['s', 95.],
            ['c', 1275.],
            ['b', 4180.]],
        columns=['name','m (MeV)']
        )

    def parse_quantum_numbers(self, s):
        """
        Parses individual quantum numbers from strings of
        the form, e.g., '0+(2++)'
        """
        ig, jpc = s.split('(')
        jpc = jpc.rstrip(')')
        # isospin
        i = ig.rstrip('+').rstrip('-')
        # g-parity
        if '+' in ig:
            g = '+'
        elif '-' in ig:
            g = '-'
        else:
            g = None
        # spin
        j = jpc.rstrip('+').rstrip('-')
        # parity
        p = jpc[1]    
        # charge conjugation
        if len(jpc) == 2:
            c = None
        elif len(jpc) == 3:
            c = jpc[2]
        else:
            raise ValueError("Cannot parse: {0}".format(s))
        return {'I':i, 'G':g, 'J':j, 'P':p, 'C':c}

    def to_lattice_units(self, m, a):
        """
        Converts to lattice units
        ma = m (MeV) a (fm) / hbarc (MeV*fm)
        """
        hbarc = 197.32997 # MeV * fm
        return m * a / hbarc

    def convert_to_MeV(self, ma, a):
        """
        Converts to MeV
        m = ma / a (fm) * hbarc (MeV*fm)
        """
        hbarc = 197.32997 # MeV * fm    
        return ma * hbarc / a
    
    def get_closest_quark(self, ma):
        mass = min(self.quarks['ma'].values, key=lambda x:abs(x-ma))
        mask = (self.quarks['ma'] == mass)
        name = self.quarks['name'][mask].item()
        return name, mass
    
    def get_quark_content(self, quark_tag, antiquark_tag):
        """ Gets the quark content, e.g., 'll' or 'cl' """
        mask_q    = (self.quarks_simulated['name_simulated'] == quark_tag)
        mask_qbar = (self.quarks_simulated['name_simulated'] == antiquark_tag)
        quark_content = "{0}{1}".format(
            self.quarks_simulated[mask_q]['name_physical'].item(),
            self.quarks_simulated[mask_qbar]['name_physical'].item()
        )
        return quark_content
        
    def get_relevant_mesons(self, J='0', P='-'): 
        # Identify the quark content of the simulated mesons
        # In a slight abuse of notation, the lowercase Ls here mean 
        # "light as opposed to heavy" and can refer to u,d,s, or l quarks.
        # The spectator quark is spectator on both legs
        content_ll = self.get_quark_content('spectator_quark_mass', 'antiquark_mass')
        content_hl = self.get_quark_content('spectator_quark_mass', 'heavy_mass')
        print('[+] Content of light-light meson: {0}'.format(content_ll))
        print('[+] Content of heavy-light meson: {0}'.format(content_hl))
        
        # Fetch mesons from PDG with matching quark content
        if (self.params['prefix_2pt'] != 'P5') or (self.params['prefix_3pt'] != 'P5-P5_S-S'):
            raise NotImplemented("Other spin-taste combinations not yet implemented.")
        mask_jp = (self.mesons['J'] == J) & (self.mesons['P'] == P)
        
        # kludge: make sure that we get, e.g., both 'sc' and 'cs' quark content
        mask_ll = ((self.mesons['quark_content'] == content_ll) |\
                   (self.mesons['quark_content'] == content_ll[::-1])) & mask_jp
        mask_hl = ((self.mesons['quark_content'] == content_hl) |\
                   (self.mesons['quark_content'] == content_hl[::-1])) & mask_jp

        mesons_ll = self.mesons[mask_ll]
        mesons_hl = self.mesons[mask_hl]

        return {'light-light:E': mesons_ll,
                'heavy-light:E': mesons_hl }        

    def get_opposite_parity_mesons(self):
        tmp = self.get_relevant_mesons(P='+')
        tmp['light-light:Eo'] = tmp.pop('light-light:E')
        tmp['heavy-light:Eo'] = tmp.pop('heavy-light:E')
        return tmp
    
    def make_prior(self, nstates, width=0.1, dE=0.5):
        """     
        Default width of 0.1 in lattice units.
        Default energy splitting of 0.5 in lattice units.
        """
        def to_vector(ptag):
            """ Converts a momentum tag to a vector, e.g., 'p000' --> [0., 0., 0.] """
            return np.array(list(ptag.lstrip('p')), dtype=float)

        def boost(E, p):
            """ Boosts a rest-frame energy by the vector p"""
            p2 = np.dot(p,p)
            E2 = E**2.
            return np.sqrt(E2 + p2)

        p = to_vector(self.params['p'])*2.0*np.pi/self.params['ns']
        if len(p) != 3:
            raise ValueError("Boosted priors expect a three-vector.")

        relevant_mesons = self.get_relevant_mesons()
        # Switch to moving frame for light-light state only
        E_ll = boost(relevant_mesons['light-light:E']['ma'].values, p)
        E_hl = relevant_mesons['heavy-light:E']['ma'].values

        partner_mesons = self.get_opposite_parity_mesons()
        Eo_ll = boost(partner_mesons['light-light:Eo']['ma'].values, p)
        Eo_hl = partner_mesons['heavy-light:Eo']['ma'].values
        
        # Initial guess, move E -> dE
        p0 = {
            'light-light:dE':  np.diff(np.insert(E_ll, 0, 0.)),
            'heavy-light:dE':  np.diff(np.insert(E_hl, 0, 0.)),
            'light-light:dEo': np.diff(np.insert(Eo_ll, 0, 0.)),
            'heavy-light:dEo': np.diff(np.insert(Eo_hl, 0, 0.)),
        }
        # Build prior using initial guesses
        prior = {}
        for tag, (n_decay, n_oscillating) in nstates.iteritems():
            # Handle the decaying states first
            pname = "{0}:dE".format(tag)
            prior[pname] = np.zeros(n_decay, dtype=object)
            for idx in range(n_decay):
                try:
                    # Use initial guess from the PDG, when present
                    mean = p0[pname][idx]
                except IndexError:
                    # Use alternative guess otherwise
                    mean = dE
                prior[pname][idx] = gv.gvar(mean, width)

            # Next handle the oscillating states
            pname = "{0}:dEo".format(tag)
            prior[pname] = np.zeros(n_oscillating, dtype=object)
            for idx in range(n_oscillating):
                try:
                    # Use initial guess from the PDG, when present
                    mean = p0[pname][idx]
                except IndexError:
                    # Use alternative guess otherwise
                    mean = dE
                prior[pname][idx] = gv.gvar(mean, width)
                
        return prior                