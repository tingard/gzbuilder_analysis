import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from . import fitting, pipeline, oo, metric
from . import _vprint

_imshow_kwargs = dict(origin='lower', cmap='gray_r')


def _figure_wrapper(f):
    def _func(*args, outfile=None, figsize=(8, 8), **kwargs):
        plt.figure(figsize=figsize)
        kw = f(*args, **kwargs)
        if outfile is not None:
            save_kwargs = kw if kw is not None else {}
            plt.savefig(outfile, **save_kwargs)
        plt.close()
    return _func


class FigureWrapper():
    def __init__(self, func):
        self.func = func
        self.sets = []

    def add(self, *args, **kwargs):
        self.sets.append([args, kwargs])

    def __len__(self):
        return len(self.sets)

    def __call__(self, *args, ax=None, outfile=None,
                 clear=True, **kwargs):
        if len(self.sets) < 1:
            raise IndexError('No datasets added')
        figsize = kwargs.pop('figsize', (7 * len(self.sets), 8))
        projection = kwargs.get('projection', None)
        if ax is None:
            plt.figure(figsize=figsize)
        for i, set in enumerate(self.sets):
            if ax is not None:
                plot_axis = ax[i]
                plt.sca(ax[i])
            else:
                plt.subplot(1, len(self.sets), i + 1, projection=projection)
                plt.title('Arm {}'.format(i))
                plot_axis = plt.gca()
            kw = self.func(*set[0], **set[1], **kwargs)
            if i > 0:
                plt.ylabel('')
                try:
                    plot_axis.get_legend().remove()
                except AttributeError:
                    pass
        if outfile is not None:
            save_kwargs = kw if kw is not None else {}
            plt.savefig(outfile, **save_kwargs)
        if clear:
            self.sets = []


@_figure_wrapper
def plot_deprojected_array(image_array, deprojected_array, phi, ba):
    plt.subplot(121)
    plt.imshow(image_array, origin='lower', cmap='gray_r')
    plt.gca().add_patch(Ellipse(
        xy=(image_array.shape[0]/2, image_array.shape[1]/2),
        width=128,
        height=128 * ba,
        angle=phi,
        ec='C0', fc='none', linewidth=5,
    ))
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(deprojected_array, origin='lower', cmap='gray_r')
    plt.axis('off')


@_figure_wrapper
def plot_drawn_arms(drawn_arms, image_arr=None, **kwargs):
    if image_arr is not None:
        plt.imshow(image_arr, **_imshow_kwargs)
    for arm in drawn_arms:
        plt.plot(*arm.T)
    plt.axis('off')
    plt.gcf().subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.tight_layout()
    return dict(bbox_inches='tight', pad_inches=0)


@_figure_wrapper
def plot_clustered_points(drawn_arms, labels, image_arr=None, **kwargs):
    if image_arr is not None:
        plt.imshow(image_arr, **_imshow_kwargs)

    for l in np.unique(labels):
        for arm in drawn_arms[labels == l]:
            if l < 0:
                continue
            plt.plot(*arm.T, '.', c='C{}'.format(l % 10), markersize=2,
                     alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    return dict(bbox_inches='tight', pad_inches=0)


@FigureWrapper
def make_cleaning_plot(R, t, mask, projection=None, **kwargs):
    plt.plot(t[~mask], R[~mask], 'r.', label='Points removed',
             markersize=4, alpha=0.5)
    plt.plot(t[mask], R[mask], '.', label='Points kept',
             markersize=4, alpha=0.5)
    if projection != 'polar':
        plt.xlabel(r'$\theta$ (unwrapped)')
        plt.ylabel('Distance from center')
    else:
        plt.ylim(0, max(R))

    plt.legend()


@FigureWrapper
def make_point_weight_plot(R, weights, **kwargs):
    a = np.argsort(R.reshape(-1))
    plt.plot(R.reshape(-1)[a], weights[a])
    plt.xlabel('Distance from center')
    plt.ylabel('Point weight')


@FigureWrapper
def make_fit_plot(R, t, fits, projection=None, **kwargs):
    x, y = (t, R)  # if projection == 'polar' else (R, t)
    plt.plot(x, y, '.', markersize=3, alpha=0.4)
    for name, fit in fits:
        plt.plot(*fit.T, label=name)
    if projection != 'polar':
        plt.xlabel(r'$\theta$ (unwrapped)')
        plt.ylabel('Distance from center')
    else:
        plt.ylim(0, max(R))

    plt.legend()


@FigureWrapper
def make_fit_comparison_plot(data, **kwargs):
    plt.gcf().suptitle('Result of Group K-fold cross validation model testing')
    d = list(data.items())
    scores = [i[1] for i in d]
    plt.boxplot(scores, showmeans=True, showfliers=False)
    r = ('poly_spiral_([0-9]+)', 'log_spiral')
    repl = (r'Polynomial (k=\1)', 'Log Spiral')
    labels = [re.sub(r[0], repl[0], i[0]) for i in d]
    labels = [re.sub(r[1], repl[1], s) for s in labels]
    plt.xticks(range(1, len(d) + 1), labels, rotation='vertical')
    plt.xlabel('Model')
    plt.ylabel(r'Negative median absolute error')
    plt.gcf().subplots_adjust(bottom=0.4)


def make_pipeline_plots(drawn_arms, *args, phi=0, ba=1, image_size=512,
                        distances=None, clean_points=False,
                        image_arr=None, deprojected_array=None, file_loc=None,
                        verbose=False, **kwargs):
    if distances is None:
        distances = metric.calculate_distance_matrix(drawn_arms)
    p = oo.Pipeline(drawn_arms, phi=phi, ba=ba, image_size=image_size,
                 distances=distances)
    arms = [
        p.get_arm(i, clean_points=clean_points)
        for i in range(max(p.db.labels_) + 1)
    ]

    if image_arr is not None and deprojected_array is not None:
        plot_deprojected_array(
            image_arr, deprojected_array,
            kwargs.get('phi', 0), kwargs.get('ba', 0),
            outfile='{}/image_deprojection'.format(file_loc),
            figsize=(16, 9)
        )
    plot_drawn_arms(drawn_arms, outfile='{}/drawn_arms'.format(file_loc),
                    image_arr=image_arr)
    plot_clustered_points(drawn_arms, p.db.labels_,
                          outfile='{}/arm_clusters'.format(file_loc),
                          image_arr=image_arr)
    for arm in arms:
        make_cleaning_plot.add(
            arm.R_all, arm.t_all_unwrapped, arm.outlier_mask
        )
        make_point_weight_plot.add(arm.R, arm.point_weights)

        # Calculate the models for plotting
        models, scores = arm.fit_polynomials()
        fits = [
            ('Polynomial (degree {})'.format(i), models['poly_spiral_{}'.format(i)])
            for i in range(1, 6)
        ]

        make_fit_plot.add(arm.R, arm.t, fits)
        make_fit_comparison_plot.add(scores)

    # plot all the cached data
    try:
        make_cleaning_plot(outfile='{}/cleaning'.format(file_loc), clear=False)
        plt.close()
        make_cleaning_plot(outfile='{}/cleaning_polar'.format(file_loc),
                           projection='polar', clear=True)
        plt.close()
        make_point_weight_plot(outfile='{}/point_weights'.format(file_loc),
                               clear=True)
        plt.close()
        make_fit_plot(outfile='{}/fits'.format(file_loc), clear=False)
        plt.close()
        make_fit_plot(outfile='{}/polar_fits'.format(file_loc),
                      projection='polar', clear=True)
        plt.close()
        make_fit_comparison_plot(outfile='{}/fit_comparison'.format(file_loc),
                                 clear=True)
        plt.close()

    except IndexError:
        print('No arm clusters found')



def combine_plots(plot_folder, outfile=None, title=None, im_list=[
    'image_deprojection.png',
    'cleaning.png',
    'fits.png',
    'polar_fits.png',
    'fit_comparison.png'
]):
    loc = os.path.abspath(plot_folder)
    try:
        images = [Image.open('{}/{}'.format(loc, im)) for im in im_list]
    except FileNotFoundError as e:
        print(e)
        return
    out_width = max(i.width for i in images)
    out_height = sum(i.height for i in images)
    montage = Image.new(mode='RGB', size=(out_width, out_height), color='#ffffff')
    cursor = 0
    for image in images:
        montage.paste(
            image,
            box=(int((montage.width - image.width) / 2), cursor)
        )
        cursor += image.height
    if title is not None:
        draw = ImageDraw.Draw(montage)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 64)
        w, h = font.getsize(title)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text(((out_width - w)/2, 20), title, (0, 0, 0), font=font)

    montage.save((loc + '/combined.png' if outfile is None else outfile))
