import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

def reclassify_cell_cycle(adata, cutoff):
    # default phase is S
    scores = adata.obs[['S_score', 'G2M_score']]
    phase = pd.Series('S', index=scores.index)

    # if G2M is higher than S, it's G2
    phase[scores['G2M_score'] > scores['S_score']] = 'G2M'

    # if all scores are below cutoff, it's G1...
    phase[np.all(scores < cutoff, axis=1)] = 'G1'

    adata.obs['phase_c{}'.format(cutoff)] = phase


# Functions to clean axes or subplots
def flatten_list(lst):
    """A helper function to make an flat iteratable list out of a single element or a nested list"""
    #Convert lst to a single flat and iteratable list
    if type(lst)==np.ndarray:
        lst = lst.tolist()
    
    if type(lst)!=list:
        lst = [lst]
    elif len(lst)==1:
        lst = lst[0]
    elif not all([type(l)==list for l in lst]):
        lst = [[l] if type(l)!=list else l for l in lst]
        lst = [l for sublist in lst for l in sublist]
    else:
        lst = [l for sublist in lst for l in sublist]
        
    return lst

def clean_axis(axes, remove_borders = True, remove_ticks = True, remove_axis_labels = True):
    """
    Function to remove axis elements from subplots.
    
    axes: either an individual subplot, a list of individual subplots or mix of multiple subplots in a nested list.
    """
    axes = flatten_list(axes)
    print(axes)
    for ax in axes:
        if remove_borders:
            for loc in ['left','right','top','bottom']:
                ax.spines[loc].set_visible(False)
        if remove_ticks:
            for side in ['x','y']:
                ax.tick_params(
                    axis=side,  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    left=False,
                    labelbottom=False,
                    labelleft=False,
                    labeltop = False)
                
        if remove_axis_labels:
            ax.set_ylabel('')
            ax.set_xlabel('')

def get_ax(axes, ix, ncols):
    """
    Helper function to return the correct axis
    """
    if len(axes.shape) > 1:
        return axes[ix // ncols, ix % ncols]
    else:
        return axes[ix]

def clean_subplots(ix, axes, ncols, keep_lone_xticklabels = True, **kwargs):
    """
    Function to clean up (remove borders, ticks and labels) from subplots. When ix is the last index, then the function cleans up the unused suplots.
    
    ix: last index that was used in the for loop to calculate the next ax
    axes: axes object from subplots initialisation
    ncols: number of columns on the subplot
    kwargs: key-word arguments that are passed along to function clean_axis. Can be: remove_borders = True, remove_ticks = True, remove_axis_labels = True
    """
    # get total lenght of axes (total number of subplots)
    axes_lenght = 1
    for x in axes.shape:
        axes_lenght = axes_lenght * x
    
    # iterate over all remaining plots, while ix is smaller than total number of plots, run the code
    while (ix+1) < axes_lenght: 
        ix+=1 # take the next subplot, since ix corresponds to the last plotted subplot
        ax = get_ax(axes, ix, ncols)
        
        clean_axis(ax, **kwargs)
        
        if keep_lone_xticklabels:
            last_row_ix = ix-ncols
            ax = get_ax(axes, last_row_ix, ncols)
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=True,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                left=False,
                labelbottom=True)


def get_regression_line_cutoff(adata, gene, expr_cutoff, pseudotime_col='dpt_pseudotime', 
                               use_raw=True, regression_step_size=0.0001, regressionLineOrder=5,
                               round_decimals = 3):
    """
    Return the X coordinate where gene expression regression line intersects specified expression cutoff
    """
    ordered_pseudotime = adata.obs[pseudotime_col].sort_values().values
    ordered_cells = adata.obs[pseudotime_col].sort_values().index
    ordered_index = [list(adata.obs_names).index(x) for x in ordered_cells]

    expression_dpt = []

    if all([use_raw, gene in adata.raw.var_names]):
            gene_ix = list(adata.raw.var_names).index(gene)

            if type(adata.raw.X) == np.ndarray:
                for ix in ordered_index:
                    expression_dpt.append(adata.raw.X[ix][gene_ix])
            else:
                tmp = pd.DataFrame.sparse.from_spmatrix(scipy.sparse.csr_matrix(adata.raw.X)).fillna(0)
                for ix in ordered_index:
                    expression_dpt.append(tmp[gene_ix].iloc[ix])
    elif gene in adata.var_names:
            gene_ix = list(adata.var_names).index(gene)

            if type(adata.X) == np.ndarray:
                for ix in ordered_index:
                    expression_dpt.append(adata.X[ix][gene_ix])
            else:
                tmp = pd.DataFrame.sparse.from_spmatrix(scipy.sparse.csr_matrix(adata.raw.X)).fillna(0)
                for ix in ordered_index:
                    expression_dpt.append(tmp[gene_ix].iloc[ix])

    xnew = np.arange(0, 1+regression_step_size, regression_step_size)
    regressionLine = np.polyfit(ordered_pseudotime, expression_dpt, regressionLineOrder)
    p = np.poly1d(regressionLine)
    
    return xnew[np.where(np.round(p(xnew), decimals = round_decimals) == expr_cutoff)[0]]
    

        
def initialize_subplots(groups_to_plot, ncols = 3, figsize_multiplier = (7,5), gridspec_kw = None, figsize = None, print_help = True, **fig_kw):
    if type(groups_to_plot)==list:
        total = len(groups_to_plot)
    else:
        total = groups_to_plot
    nrows = int(np.ceil(total/ncols))
    if not figsize:
        figsize = (figsize_multiplier[0]*ncols, figsize_multiplier[1]*nrows)
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize, gridspec_kw = gridspec_kw, **fig_kw)
    if print_help:
        if nrows>1 and ncols>1:
            print('ax = axes[ix // ncols, ix % ncols]')
        else:
            print('ax = axes[ix]')
    return fig, axes

    
# Plot gene expression on pseudotime
def plot_pseudotime_gene_expression(adata, gene, color = None, cmap = 'viridis', cbar = True, 
                                    figsize = (15,5), s = 10, alpha = 1, plot_type = 'scatter', ax = None, legend = True,
                                    return_axis = False, pseudotime_col = 'dpt_pseudotime', use_raw = True,
                                    plot_regression = True, regression_step_size = 0.001, regressionLineOrder = 4, 
                                    plot_reg_expr_intercept = False, round_decimals = 3, expr_cutoff = None, print_intercept = True,
                                    **line_kwargs):
    """
    Plot gene expression along pseudotime
    
    """
    if not ax:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize = figsize)
    
    ordered_pseudotime = adata.obs[pseudotime_col].sort_values().values
    ordered_cells = adata.obs[pseudotime_col].sort_values().index
    ordered_index = [list(adata.obs_names).index(x) for x in ordered_cells]
    
    expression_dpt = []
    
    if all([use_raw, gene in adata.raw.var_names]):
        gene_ix = list(adata.raw.var_names).index(gene)
        
        if type(adata.raw.X) == np.ndarray:
            for ix in ordered_index:
                expression_dpt.append(adata.raw.X[ix][gene_ix])
        else:
            tmp = pd.DataFrame.sparse.from_spmatrix(scipy.sparse.csr_matrix(adata.raw.X)).fillna(0)
            for ix in ordered_index:
                expression_dpt.append(tmp[gene_ix].iloc[ix])
                
    elif gene in adata.var_names:
        gene_ix = list(adata.var_names).index(gene)
        
        if type(adata.X) == np.ndarray:
            for ix in ordered_index:
                expression_dpt.append(adata.X[ix][gene_ix])
        else:
            tmp = pd.DataFrame.sparse.from_spmatrix(scipy.sparse.csr_matrix(adata.raw.X)).fillna(0)
            for ix in ordered_index:
                expression_dpt.append(tmp[gene_ix].iloc[ix])
                
    
    elif gene in adata.obs.columns:
        expression_dpt = adata.obs[gene].loc[ordered_cells]
    
    if not color:
        c_dict = {str(ix): x for ix, x in enumerate(adata.uns['louvain_colors'])}
        color = [c_dict[x] for x in adata.obs['louvain'].loc[ordered_cells]]
    elif color == 'pseudotime':
        color = ordered_pseudotime
    elif color+'_colors' in adata.uns.keys():
        c_dict = {x:adata.uns[color+'_colors'][ix] for ix, x in enumerate(adata.obs[color].cat.categories)}
        color = [c_dict[x] for x in adata.obs[color].loc[ordered_cells]]
        if legend:
            legend_handles = [mpl.patches.Patch(color = c_dict[key], label = key) for key in c_dict.keys()]
    elif color in adata.obs.columns:
        color = adata.obs[color].loc[ordered_cells]
    else:
        color = color
    
    if plot_type == 'scatter':
        ax.scatter(x = ordered_pseudotime, y = expression_dpt, c = color, cmap = cmap, s = s, alpha = alpha)
    elif plot_type == 'bar':
        ax.bar(x = ordered_pseudotime, height = expression_dpt, color = color, width = 0.001, alpha = alpha)
    
    if legend:
        ax.legend(handles = legend_handles)
    
    ax.set_xlim([-0.01,1.01])
    ax.set_xticks([])
    ax.set_ylabel(gene)
    
    if plot_regression:
        xnew = np.arange(0, 1+regression_step_size, regression_step_size)
        regressionLine = np.polyfit(ordered_pseudotime, expression_dpt, regressionLineOrder)
        p = np.poly1d(regressionLine)
        ax.plot(xnew, p(xnew), **line_kwargs)
        
        if plot_reg_expr_intercept:
            intercept = xnew[np.where(np.round(p(xnew), decimals = round_decimals) == expr_cutoff)[0]]
            ax.axvline(intercept[0])
            ax.axhline(expr_cutoff)
            if print_intercept:
                ax.text(intercept[0], np.floor(max(expression_dpt)), s = '{:.2f}'.format(intercept[0]))
        
    if cbar:
        norm = mpl.colors.Normalize(vmin=min(ordered_pseudotime), vmax=max(ordered_pseudotime))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        plt.colorbar(sm, orientation = 'horizontal', ax = ax, ticks = [], shrink = 1, aspect = 80, pad = 0.01)
        
    if return_axis:
        return ax