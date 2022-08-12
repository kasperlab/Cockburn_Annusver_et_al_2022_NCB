import numpy as np
import scipy
import pandas as pd
import scanpy as sc
import seaborn as sbn
from scipy.spatial import ConvexHull
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_percentage_df(adata, marker, marker_column, groupby_columns):
    #Function to get percentages and counts of a specified marker gene that's already been classified
    """
    adata - anndata object
    marker - class of cells to find percentages for. Currently works only for positive-negative classification
    marker_column - adata.obs column name that has marker classification
    groupby_columns - adata.obs columns to use to group data. Currently works only with 2 columns
    
    returns a dataframe with multiindex of groups and data about positive marker percentages, positive marker counts and toatl counts.
    """
    # Find total counts
    tmp_total_count_df = adata.obs.pivot_table(index = groupby_columns[0], columns = groupby_columns[1], values = marker_column, fill_value = 0, aggfunc = 'count').unstack()
    
    # Find counts of positive cells
    tmp_data = adata[adata.obs[marker_column]==marker]
    tmp_marker_count_df = tmp_data.obs.pivot_table(index = groupby_columns[0], columns = groupby_columns[1], values = marker_column, fill_value = 0, aggfunc = 'count').unstack()
    
    # Calculate percentages, make a combined dataframe
    tmp_percentage_df = (tmp_marker_count_df.div(tmp_total_count_df)*100).fillna(0).to_frame().rename(columns={0:'Percentage'})
    tmp_percentage_df['{} Cells'.format(marker)] = tmp_marker_count_df.values
    tmp_percentage_df['Total Cells'] = tmp_total_count_df.values
    
    return tmp_percentage_df

def classify_marker_expression(adata, marker, cutoff=0, use_raw = True, use_layers = False, layer_name = None, return_percentage_df = True, groupby_columns = ['cl_2nd_EPI','phase']):
    #Classify cells into positive and negative according to gene expression
    """
    adata - anndata object
    marker - gene to be used for classification
    cutoff - gene expression greater than cutoff is considered positive
    use_raw - whether to use the .raw attribute of adata. Default: True
    use_layers - whether to use .layers attribute of adata. Default: False
    layer_name - which layer name to use for layer attribute
    return_percentage_df - calculate percentages of classified cells and return a DataFrame
    groupby_columns - columns to pass on to get_percentage_df
    """
    
    if all([use_raw, use_layers]):
        print('Cannot use both raw and layers attributes together, select only one of them.')

    if use_raw:
        ix = np.where(adata.raw.var_names == marker)[0]
    else:
        ix = np.where(adata.var_names == marker)[0]
    
    if use_layers:
        tmp_mtx = adata.layers[layer_name][:,ix]
        if type(tmp_mtx)==np.matrix:
            gene_array = np.array(tmp_mtx.flatten())[0]
        elif type(tmp_mtx)==np.ndarray:
            gene_array = np.array(tmp_mtx.flatten())
        else:
            gene_array = np.array(tmp_mtx.todense().flatten())[0]
    elif use_raw:
        tmp_mtx = adata.raw.X[:,ix]
        if type(tmp_mtx)==np.matrix:
            gene_array = np.array(tmp_mtx.flatten())[0]
        elif type(tmp_mtx)==np.ndarray:
            gene_array = np.array(tmp_mtx.flatten())
        else:
            gene_array = np.array(tmp_mtx.todense().flatten())[0]
    else:
        tmp_mtx = adata.X[:,ix]
        if type(tmp_mtx)==np.matrix:
            gene_array = np.array(tmp_mtx.flatten())[0]
        elif type(tmp_mtx)==np.ndarray:
            gene_array = np.array(tmp_mtx.flatten())
        else:
            gene_array = np.array(tmp_mtx.todense().flatten())[0]
    
    adata.obs['{} Class'.format(marker)] = ['{}+'.format(marker) if x else '{}-'.format(marker) for x in gene_array > cutoff]
    
    if return_percentage_df:
        return get_percentage_df(adata = adata, marker = '{}+'.format(marker), marker_column = '{} Class'.format(marker), groupby_columns = groupby_columns)


def reclassify_cell_cycle(adata, cutoff):
    # default phase is S
    scores = adata.obs[['S_score', 'G2M_score']]
    phase = pd.Series('S', index=scores.index)

    # if G2M is higher than S, it's G2
    phase[scores['G2M_score'] > scores['S_score']] = 'G2M'

    # if all scores are below cutoff, it's G1...
    phase[np.all(scores < cutoff, axis=1)] = 'G1'

    adata.obs['phase_c{}'.format(cutoff)] = phase

################################
####   Plotting functions   ####
################################

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



def plot_umap_with_legends(adata, color = 'leiden', ax = None, s = 200, frameon = False, legend_fontsize = 14, legend_fontoutline = 2, side_legend = 'right margin'):
    """
    Plot umap with legends both on the side and on the clusters
    """
    if not ax:
        fig, ax = plt.subplots(figsize = (8,4))
    sc.pl.umap(adata, color = color, ax = ax, show = False, s = s, frameon = frameon, legend_loc = side_legend)
    sc.pl.umap(adata, color = color, ax = ax, show = False, s = 0, legend_loc = 'on data', frameon = frameon, legend_fontsize = legend_fontsize, legend_fontoutline = legend_fontoutline)            

    
# Plot heatmap
def plot_group_composition_heatmap(dataframe, groupby_name, count_group_name, title = 'long',
                                   observed = True, percentage = True, transpose = False, order_columns = None,
                                   ax = None, annot = True, fmt = '3.0f', **kwargs):
    """
    Heatmap plotting function for comparing the composition of 2 groups in a dataframe.
    
    dataframe: dataframe containing the columns with observation data
    groupby_name: column to use for the groupby function
    count_group_name: column to use for value_counts function
    title: what to plot as the subplot titles. Either 'long', 'short' or custom string
    observed: if groupby_name column is categorical and includes not observed values, then observed = False will result in an error (default: True)
    percentage: whether to plot the normalized fractions of the counts
    transpose: whether to transpose the heatmap
    order_columns: a list of heatmap column names specifying their order
    ax: axis to plot on
    annot: whether to annotate the heatmap (default: True)
    fmt: how to format the annotations on the heatmap (default: 3.0f - shows 3 digits with no decimals)
    **kwargs: additional key-word arguments passed on to heatmap (for example vmax and vmin to specify ranges for colormap)
    """
    df = (dataframe.groupby(by = groupby_name, observed = True)[count_group_name]
          .value_counts(normalize = percentage)*(100 if percentage else 1))
    
    col_name = 'Percentage' if percentage else 'Counts'
    df = (df.to_frame(col_name)
          .reset_index()
          .pivot(index = groupby_name, columns = count_group_name, values = col_name)
          .fillna(0))
    
    if transpose: 
        df = df.T
    if order_columns:
        df = df[order_columns]
    
    if title == 'long':
        title = 'Composition ({}) of "{}" cells in "{}"'.format('%' if percentage else 'counts', count_group_name, groupby_name)
    elif title == 'short':
        title = '{} in {} ({})'.format(count_group_name, groupby_name, '%' if percentage else 'counts')
        
    g = sbn.heatmap(df, annot = annot, fmt = fmt, ax = ax, **kwargs)
    g.set_title(title)


# Plot gene expression on pseudotime
def plot_pseudotime_gene_expression_old(adata, gene, c = None, cmap = 'viridis', cbar = True, 
                                    figsize = (15,5), s = 10, plot_type = 'scatter', 
                                    return_axis = False, pseudotime_col = 'dpt_pseudotime', use_raw = True):
    """
    Plot gene expression along pseudotime
    
    """
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
            tmp = pd.SparseDataFrame(adata.raw.X).fillna(0)
            for ix in ordered_index:
                expression_dpt.append(tmp[gene_ix].iloc[ix])
                
    elif gene in adata.var_names:
        gene_ix = list(adata.var_names).index(gene)
        
        if type(adata.X) == np.ndarray:
            for ix in ordered_index:
                expression_dpt.append(adata.X[ix][gene_ix])
        else:
            tmp = pd.SparseDataFrame(adata.X).fillna(0)
            for ix in ordered_index:
                expression_dpt.append(tmp[gene_ix].iloc[ix])
                
    elif gene in adata.obs.columns:
        expression_dpt = adata.obs[gene].loc[ordered_cells]
    
    if not c:
        c_dict = {str(ix): x for ix, x in enumerate(adata.uns['louvain_colors'])}
        c = [c_dict[x] for x in adata.obs['louvain'].loc[ordered_cells]]
    elif c == 'pseudotime':
        c = ordered_pseudotime
    elif c+'_colors' in adata.uns.keys():
        c_dict = {x:sorted(adata.uns[c+'_colors'])[ix] for ix, x in enumerate(sorted(set(adata.obs[c])))}
        c = [c_dict[x] for x in adata.obs[c].loc[ordered_cells]]
    else:
        c = adata.obs[c].loc[ordered_cells]
        
    if plot_type == 'scatter':
        ax.scatter(x = ordered_pseudotime, y = expression_dpt, c = c, cmap = cmap, s = s)
    elif plot_type == 'bar':
        ax.bar(x = ordered_pseudotime, height = expression_dpt, color = c, width = 0.001)
    
    ax.set_xlim([-0.01,1.01])
    ax.set_xticks([])
    ax.set_ylabel(gene)
    
    if cbar:
        norm = mpl.colors.Normalize(vmin=min(ordered_pseudotime), vmax=max(ordered_pseudotime))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        plt.colorbar(sm, orientation = 'horizontal', ax = ax, ticks = [], shrink = 1, aspect = 80, pad = 0.01)
        
    if return_axis:
        return ax


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
    

# Plot population area
def plot_points_area(adata, group, basis = 'umap', ax = None, z_score = 2, alpha = 0.4, fill = True, legend = None, colors = None, **poly_kwargs):
    """
    Plot an approximation of the central area covered by a population (i.e., main area for basal cells).
    
    adata: anndata object
    group: adata.obs column name to use for plotting
    basis: which embedding to use for plotting (default: umap)
    ax: axis to plot on. If not defined, then it creates one
    z_score: how many outliers to remove depending on their z-score. This helps to constrain the area to where main datapoints are. Otherwise a single outlier can make the area a lot bigger. (default: 2)
    alpha: transparency of the plotted area. Only useful if fill=True (default: 0.4)
    fill: whether to color the inside of the area (fill=True) or to just have an outline (fill=False)
    legend: add group legend to the plot.
    colors: define colors for area plotting. If None, then colors are taken from adata.uns
    poly_kwargs: keyword arguments passed on to polygon creation. These include linestyle and linewidth
    """
    if not ax:
        fig, ax = plt.subplots()
    
    for ix, clust in enumerate(adata.obs[group].cat.categories):
        points = adata.obsm['X_{}'.format(basis)][np.where(adata.obs[group]==clust)[0]]
        if colors:
            c = colors[ix]
        else:
            c = adata.uns['{}_colors'.format(group)][ix]

        x_z = scipy.stats.zscore(points[:,0])
        y_z = scipy.stats.zscore(points[:,1])
        indeces = np.where([all(x) for x in zip(abs(x_z)<z_score, abs(y_z)<z_score)])[0]

        points = points[indeces]

        hull = ConvexHull(points)
        poly = plt.Polygon(points[hull.vertices], color = c, alpha = alpha, fill = fill, **poly_kwargs)
        
        ax.add_patch(poly)
    
    if legend:
        ax.legend(labels = adata.obs[group].cat.categories)
        
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

def convert_rgb_to_hex(rgb = None, adata = None, color_name = None):
    if rgb:
        return ['#%02x%02x%02x' % (int(x[0]*255), int(x[1]*255), int(x[2]*255)) for x in rgb]
    if adata:
        return ['#%02x%02x%02x' % (int(x[0]*255), int(x[1]*255), int(x[2]*255)) for x in adata.uns['{}_colors'.format(color_name)]]
    

    
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
    
def get_marker_gene_list(adata, by = 'pvals_adj', genes_as_index = True):
    """
    Helper function that returns for each differentially expressed gene 
    the minimum p_value (or fold change or score) and the corresponding 
    cluster information and other associated values.
    
    adata: AnnData object where adata.uns['rank_genes_groups'] contains the 
           following keys: dict_keys(['params', 'scores', 'names', 'logfoldchanges', 'pvals', 'pvals_adj'])
    by: which key to use for filtering - p-value keys filter by minimum value, logfoldchange and scores filter by maximum value
    genes_as_index: wheter to set genes as the index in the output dataframe
    """
    #Create the dataframe objects for each key
    pvals = pd.DataFrame(adata.uns['rank_genes_groups']['pvals'])
    pvals_adj = pd.DataFrame(adata.uns['rank_genes_groups']['pvals_adj'])
    gene_names = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    foldchanges = pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges'])
    scores = pd.DataFrame(adata.uns['rank_genes_groups']['scores'])
    
    #Unpivot the dataframes with the pandas.melt function
    #Add the appropriate column to the correct dataframe
    df_melted = pd.melt(pvals, var_name="cluster", value_name="pvals")    
    df_melted['logfoldchanges'] = pd.melt(foldchanges, var_name="cluster", value_name="foldchanges")['foldchanges']
    df_melted['scores'] = pd.melt(scores, var_name="cluster", value_name="scores")['scores']
    df_melted['gene'] = pd.melt(gene_names, var_name="cluster", value_name="gene")['gene']
    df_melted['pvals_adj'] = pd.melt(pvals_adj, var_name="cluster", value_name="pvals_adj")['pvals_adj']
    
    #Get the lowest or highest values, depending on the column defined in 'by'
    if by in ['pvals_adj', 'pvals']:
        out_df = df_melted.iloc[df_melted.groupby(by=["gene"])[by].idxmin()]
    elif by in ['logfoldchanges', 'scores']:
        out_df = df_melted.iloc[df_melted.groupby(by=["gene"])[by].idxmax()]
    else:
        raise NameError("Valid options for 'by' parameter are: ['pvals_adj','pvals','logfoldchanges','scores']")
    
    #Set genes to be the index of the output dataframe (keep genes column)
    if genes_as_index:
        out_df = out_df.set_index('gene', drop = False).rename_axis(None)
    
    return out_df


def get_gene_expression_different_groups(adata, gene, group1_column, group1_values, group2_column, group2_values, use_raw = True):
    """
    group1/2_column: which column to use for annotation information
    group1/2_values: subset of values to include from respective annotation
    gene: which gene to include
    use_raw: whether to use raw counts or not
    """
    
    def get_group_gene_expression_helper(adata, gene, group_column, group_values, use_raw = True):
        """
        Helper function to return gene expression of cells belonging to the respective group.
        Error handling should take care of both dense and sparse count matrixes
        """
        if use_raw:
            try:
                gene_counts = adata.raw[:, gene].X.flatten()
            except AttributeError:
                gene_counts = adata.raw[:, gene].X.toarray().flatten()
        else:
            try:
                gene_counts = adata[:, gene].X.flatten()
            except AttributeError:
                gene_counts = adata[:, gene].X.toarray().flatten()

        grouping = adata.obs[group_column].values
        grouping = ['{}\n{}'.format(group_column.capitalize(), x) for x in grouping]

        return pd.DataFrame(np.column_stack([grouping, gene_counts]), columns = ['Grouping', gene])
    
    #In case group values are not lists, convert them to a list
    if type(group1_values)!=list:
        group1_values = [group1_values]
    if type(group2_values)!=list:
        group2_values = [group2_values]
    
    #Use helper function to obtain group specific dataframes
    df1 = get_group_gene_expression_helper(adata[adata.obs[group1_column].isin(group1_values)], gene, group1_column, group1_values, use_raw = use_raw)
    df2 = get_group_gene_expression_helper(adata[adata.obs[group2_column].isin(group2_values)], gene, group2_column, group2_values, use_raw = use_raw)
    
    #Concatenate dataframes and make sure gene expression values are float
    df = pd.concat([df1, df2], axis = 0, ignore_index = True)
    df[gene] = df[gene].astype(float)
    return df

def calc_density(x: np.ndarray, y: np.ndarray):
    """\
    Function to calculate the density of cells in an embedding.
    Taken from scanpy embedding density calculation
    """
    from scipy.stats import gaussian_kde

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    min_z = np.min(z)
    max_z = np.max(z)

    # Scale between 0 and 1
    scaled_z = (z - min_z) / (max_z - min_z)

    return scaled_z

def calculate_jaccard(df, n_genes):
    """
    Calculate jaccard similarity matrix between groups of genes.
    Input must be a dataframe with groups in columns and genes as elements

    """

    def get_genes(df, col, n_genes):
        #Helper function to get the genes from the dataframe and remove NAs
        return df[col].dropna().iloc[:n_genes].values

    def calculate_jaccard_index(df, col1, col2, n_genes):
        #Helper function to calculate Jaccard index between each group
        a = get_genes(df, col1, n_genes)
        b = get_genes(df, col2, n_genes)
        return len(np.intersect1d(a,b))/len(np.union1d(a,b))

    res = {}
    cols = df.columns
    for col1 in cols:
        res[col1] = [calculate_jaccard_index(df, col1, col2, n_genes) for col2 in cols]
        
    jac_df = pd.DataFrame.from_dict(res)
    jac_df.index = cols
    
    return jac_df


def get_highest_expressed_genes(adata, n_top):
    """
    Code adapted from scanpy sc.pl.highest_expr_genes() function
    Find the names of top highest expressed genes in the dataset
    """
    from scanpy.preprocessing._normalization import normalize_total
    from scipy.sparse import issparse

    norm_dict = normalize_total(adata, target_sum=100, inplace=False)
    
    if issparse(norm_dict['X']):
        mean_percent = norm_dict['X'].mean(axis=0).A1
        top_idx = np.argsort(mean_percent)[::-1][:n_top]
    else:
        mean_percent = norm_dict['X'].mean(axis=0)
        top_idx = np.argsort(mean_percent)[::-1][:n_top]
    
    return adata.var_names[top_idx]


def find_unique_genes(adata, clustering, run_parallel = True, n_threads = 200, alpha = 0.05, expression_cutoff = 0.5, percentile = 0.9, estimator = 'percentile', return_bottom = False, bottom_estimator = 'percentile', bottom_percentile = 0.1):
    """
    Find uniquely expressed genes for each group in the dataset
    Can run in parallel with ThreadPool function from multiprocessing
    
    n_threads : how many threads to use for parallel processing
    alpha : significanse value for cutoff
    expression_cutoff : lowest expression level to consider for a marker gene
    percentile: percentile to use when estimator is 'percentile'
    return_bottom: find negative marker genes for each population (lowly expressed compared to all other clusters)
    bottom_estimator: estimator to use for finding bottom genes
    bottom_percentile: percintile to use if bottom_estimator is 'percentile'
    
    Returns
    sec_high: pandas DataFrame of mean expression per cluster for each gene
    genes: list of significantly different genes between datasets
    top_genes: dataframe of unique genes for each cluster
    bottom_genes: if return_bottom is selected, return a dataframe of negative markers
    """
    # Get the count matrix and use cluster identity as index
    
    # Group the counts for each gene in each cluster by the chosen statistic
    if estimator == 'mean':
        grouped_counts = pd.DataFrame(data = adata.raw.X.toarray(), index = adata.obs[clustering], columns = adata.raw.var_names).reset_index().groupby(clustering).mean()
    elif estimator == 'median':
        grouped_counts = pd.DataFrame(data = adata.raw.X.toarray(), index = adata.obs[clustering], columns = adata.raw.var_names).reset_index().groupby(clustering).median()
    elif estimator == 'percentile':
        grouped_counts = pd.DataFrame(data = adata.raw.X.toarray(), index = adata.obs[clustering], columns = adata.raw.var_names).reset_index().groupby(clustering).quantile(percentile)
    else:
        raise NotImplementedError(f'Estimator {estimator} not implemented. Valid options are: mean, median or percentile')
    
    #Filter out genes without any counts
    grouped_counts = grouped_counts[grouped_counts.columns[grouped_counts.max()>0]]

    #Compare each value to the second highest value
    ## Add a tiny pseudocount to the second highest count in case the actual count is 0 (can't divide by 0)
    ## Take log2(x+1) of the difference
    sec_high = grouped_counts.apply(lambda x: np.log1p(x/(sorted(x)[-2]+0.000001)))

    # Find the highest and second highest expressing cluster for each gene
    order = pd.DataFrame(np.argsort(grouped_counts.values, axis = 0)[-2:, :][::-1], columns = grouped_counts.columns, index = ['Highest', 'Second'])


    if run_parallel == True:
        
        from multiprocessing.pool import ThreadPool

        def calculate_significance(adata, gene, vals, clustering):
            high_x = adata.raw[adata.obs[clustering]==str(vals[0])][:, gene].X.toarray().flatten()
            sec_x = adata.raw[adata.obs[clustering]==str(vals[1])][:, gene].X.toarray().flatten()
            stat, p = scipy.stats.ttest_ind(high_x, sec_x)
            return(f'{gene}_{p}')
        
        pool = ThreadPool(n_threads)
        multi_res = pool.starmap(calculate_significance, [[adata, g, vals, clustering] for g, vals in order.items()])
        pool.close()

        genes, res = zip(*[x.split('_') for x in multi_res])
        res = np.array(res, dtype = float)

    else:
        #T-test for each gene and cluster pair
        ## This can take quite a long time (took about 10 min for 28k genes)
        ## Collect p-values in res
        res = []
        for g, vals in order.items():
            high_x = adata.raw[adata.obs[clustering]==str(vals[0])][:, g].X.toarray().flatten()
            sec_x = adata.raw[adata.obs[clustering]==str(vals[1])][:, g].X.toarray().flatten()
            stat, p = scipy.stats.ttest_ind(high_x, sec_x)
            res.append(p)

    #Multiple testing correction
    ##Use statsmodels to correct
    import statsmodels
    ##Put corrected values to a Series and find significant genes
    p_vals = pd.Series(statsmodels.stats.multitest.multipletests(np.array(res), alpha=alpha)[1], order.columns)
    significant_genes = p_vals[p_vals < alpha].index
    
    ##Use only significant genes that have mean expression of at least the expression_cutoff in any cluster
    genes = grouped_counts.columns[grouped_counts.max()>expression_cutoff].intersection(significant_genes)
    
    #Make a dataframe with top genes for each cluster
    top_genes = pd.DataFrame.from_dict({x[0]: x[1].sort_values(ascending = False).index for x in sec_high[significant_genes].iterrows()})
    if return_bottom:
        return sec_high, genes, top_genes, find_bottom_genes(adata, sec_high, genes, estimator = bottom_estimator, percentile = bottom_percentile)
    else:
        return sec_high, genes, top_genes

def find_bottom_genes(adata, sec_high, signif_genes, estimator = 'percentile', percentile = 0.1):
    bottom_genes = {}
    if estimator not in ['mean', 'median','percentile']:
        raise NotImplementedError(f'Estimator {estimator} not implemented. Valid options are: mean, median or percentile')
    for cl in sec_high.index:
        tmp = sec_high[signif_genes]
        
        if estimator == 'mean':
            other_means_df = tmp.loc[tmp.index[~tmp.index.isin([cl])]].mean()
        elif estimator == 'median':
            other_means_df = tmp.loc[tmp.index[~tmp.index.isin([cl])]].median()
        elif estimator == 'percentile':
            other_means_df = tmp.loc[tmp.index[~tmp.index.isin([cl])]].quantile(percentile)
        
        selection = other_means_df[other_means_df>0].index
        tmp = tmp[selection]
        tmp = tmp[tmp.columns[np.argsort(other_means_df-tmp.loc[cl]).values]].T.sort_values(by = cl).index
        bottom_genes[cl] = tmp

    return pd.DataFrame.from_dict(bottom_genes)

def regress_out(a, b):
    """Regress b from a keeping a's original mean."""
    a_mean = a.mean()
    a = a - a_mean
    b = b - b.mean()
    b = np.c_[b]
    a_prime = a - b.dot(np.linalg.pinv(b).dot(a))
    return np.asarray(a_prime + a_mean).reshape(a.shape)

def annotate_deg_df(deg_df, dge_data, clustering = 'orig.ident', deg_lfc_cutoff = [0.5, 1], p_value = 0.05, inf_p = 300, pct_cutoff = 25):
    deg_df['-log10(pvals_adj)'] = -np.log10(deg_df['pvals_adj'])
    
    deg_df['-log10(pvals_adj)'] = [inf_p if x==np.inf else x for x in deg_df['-log10(pvals_adj)']]
    
    counts = pd.DataFrame(data = dge_data.raw.X.toarray(), index = dge_data.obs[clustering], columns = dge_data.raw.var_names)
    
    tmp = counts.copy()
    tmp.index = tmp.index.map(lambda x: ''.join([i for i in x if not i.isdigit()]))
    tmp = (tmp>0).reset_index().groupby(clustering).sum().div((tmp>=0).reset_index().groupby(clustering).sum())*100
    deg_df[[f'pct. {s}' for s in tmp.index]] = tmp[deg_df['names']].T.reset_index(drop = True)
    pct_columns = [f'pct. {s}' for s in tmp.index]
    
    counts = counts.reset_index(drop = False).groupby([clustering]).sum()>0
    counts.index = counts.index.map(lambda x: ''.join([i for i in x if not i.isdigit()]))
    counts = counts.reset_index(drop = False).groupby([clustering]).sum()

    
    deg_df['Samples'] = np.where(counts[deg_df['names']].sum()==6, 'All samples', 'Some samples')
    deg_df[[f'Expr. samples {s}' for s in counts.index]] = counts[deg_df['names']].T.reset_index(drop = True)
    
    for i in deg_lfc_cutoff:
        deg_df[f'DEG lfc:{i}'] = np.where(
            (deg_df['-log10(pvals_adj)']>-np.log10(p_value)) & (abs(deg_df['logfoldchanges'])>i) & (deg_df['Samples']=='All samples') & (deg_df[pct_columns[0]]>pct_cutoff) & (deg_df[pct_columns[1]]>pct_cutoff), 
            'DEG', 'Non-DEG')
        
    return deg_df

def print_padding(s, filler = '#', length = 30):
    """Print visible string with fillers surrounding it"""
    if len(s)>=length:
        length = len(s)+6
    padding = (length - len(s) - 4) // 2
    header = f'{filler*padding}  {s}  {filler*padding}'
    length = len(header)
    
    print(filler*length)
    print(header)
    print(filler*length)