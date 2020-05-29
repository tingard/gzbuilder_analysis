import os
import time
import warnings
import bz2
import shutil
import requests
import tempfile
from io import BytesIO
import json
from multiprocessing.pool import ThreadPool
from functools import wraps
from PIL import Image
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import NoOverlapError
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy import log
import sep
import montage_wrapper as montage
import gzbuilder_analysis.data.sdss_psf as sdss_psf


LOCATION_QUERY_URL = 'http://skyserver.sdss.org/dr13/en/tools/search/x_results.aspx'
FRAME_QUERY_URL = (
    'http://data.sdss.org/sas/dr13/eboss/photoObj/frames/301/'
    '{run}/{camcol}/'
    'frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2'
)
PSF_QUERY_URL = (
    'https://data.sdss.org/sas/dr14/eboss/photo/redux/'
    '{rerun}/{run}/objcs/{camcol}/'
    'psField-{run:06d}-{camcol}-{field:04d}.fit'
)

GAIN_TABLE = pd.DataFrame(
    [
        [1.62, 3.32, 4.71, 5.165, 4.745],
        [np.nan, 3.855, 4.6, 6.565, 5.155],
        [1.59, 3.845, 4.72, 4.86, 4.885],
        [1.6, 3.995, 4.76, 4.885, 4.775],
        [1.47, 4.05, 4.725, 4.64, 3.48],
        [2.17, 4.035, 4.895, 4.76, 4.69],
    ],
    index=pd.Series(np.arange(6) + 1, name='camcol'),
    columns=list('ugriz')
)

# TODO: add dark variance for other bands
DARKVAR_TABLE = pd.DataFrame(
    [
        [1.8225, 1.00, 1.3225, 1.3225, 0.81, 0.9025],
    ],
    columns=pd.Series(np.arange(6) + 1, name='camcol'),
    index=['r']
).T
DARKVAR_R = pd.Series(
    [1.8225, 1.00, 1.3225, 1.3225, 0.81, 0.9025],
    index=np.arange(6) + 1,
    name='darkvar_r'
)

REQUEST_TIMEOUT = 2


def __download_retry_wrapper(func, n_retries=3):
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        _err = None
        for i in range(n_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except requests.exceptions.Timeout as err:
                _err = err
        raise(_err)
    return wrapper_func


@__download_retry_wrapper
def get_frame_list(ra, dec, radius, limit=1000, silence=False, timeout=REQUEST_TIMEOUT):
    location_query_res = requests.get(
        LOCATION_QUERY_URL,
        params={
            'searchtool': 'Radial',
            'TaskName': 'Skyserver.Search.Radial',
            'whichphotometry': 'optical',
            'coordtype': 'equatorial',
            'ra': ra,
            'dec': dec,
            'radius': radius,
            'limit': limit,
            'format': 'json',
        },
        timeout=timeout,
    )
    if location_query_res.status_code == 200:
        try:
            frame_list = sorted(
                location_query_res.json()[0]['Rows'],
                key=lambda i: (
                    (i['ra'] - ra)**2 + (i['dec'] - dec)**2
                )
            )
            # only return the unique frames
            return pd.DataFrame(frame_list) \
                .set_index('objid') \
                .drop_duplicates(subset=['run', 'rerun', 'camcol', 'field'])
        except json.decoder.JSONDecodeError:
            if not silence:
                log.error('Could not parse returned JSON: ' + location_query_res.url)
            return []
    elif not silence:
        log.error('Could not connect to SDSS skyserver: ' + location_query_res.url)
    return []


def get_data_from_fits(f, band):
    img = f[0]
    ff = f[1]
    sky = f[2]
    allsky, xinterp, yinterp = sky.data[0]
    sky_img = interp2d(
        np.arange(allsky.shape[1]),
        np.arange(allsky.shape[0]),
        allsky,
    )(xinterp, yinterp)
    calib_img = np.tile(np.expand_dims(ff.data, 1), img.data.shape[0]).T
    dn = img.data / calib_img + sky_img
    gain = GAIN_TABLE.loc[img.header['camcol']][img.header['FILTER']]
    darkvar = DARKVAR_TABLE[band].loc[img.header['camcol']]
    nelec = dn * gain
    return {'fits': f, 'nelec': nelec, 'calib': calib_img, 'sky': sky_img,
            'gain': gain, 'darkvar': darkvar, 'frame_wcs': WCS(f[0])}


@__download_retry_wrapper
def __get_fits(url, silence=False, bz2_decompress=True, timeout=REQUEST_TIMEOUT):
    r = requests.get(url, timeout=timeout)
    if r.status_code == 200:
        try:
            if bz2_decompress:
                decompressor = bz2.BZ2Decompressor()
                data = decompressor.decompress(r.content)
            else:
                data = r.content
            return fits.open(BytesIO(data))
        except EOFError:
            if not silence:
                log.error('Could not download', url)
    return None


def __download_frame(band, frame, silence=False):
    url = FRAME_QUERY_URL.format(
        band=band,
        **frame[['run', 'rerun', 'camcol', 'field']].astype(int)
    )
    return __get_fits(url, silence=silence, bz2_decompress=True)


def __download_psf(frame, silence=False):
    url = PSF_QUERY_URL.format(
        **frame[['run', 'rerun', 'camcol', 'field']].astype(int)
    )
    return __get_fits(url, silence=silence, bz2_decompress=False)


@__download_retry_wrapper
def download_json(url, timeout=REQUEST_TIMEOUT):
    r = requests.get(url, timeout=timeout)
    if r.status_code == 200:
        return pd.Series(r.json())


@__download_retry_wrapper
def download_image(url, timeout=REQUEST_TIMEOUT):
    buffer = tempfile.SpooledTemporaryFile(max_size=1e9)
    r = requests.get(url, stream=True, timeout=timeout)
    if r.status_code == 200:
        downloaded = 0
        filesize = int(r.headers['content-length'])
        for chunk in r.iter_content():
            downloaded += len(chunk)
            buffer.write(chunk)

        buffer.seek(0)
        i = Image.open(BytesIO(buffer.read()))
    buffer.close()
    return i


def get_frame_psf_at_points(frame, wcs, ra, dec):
    coords = wcs.wcs_world2pix([[ra, dec]], 1)
    psfield = __download_psf(frame)
    bandnum = 'ugriz'.index('r') + 1
    hdu = psfield[bandnum]
    return sdss_psf.sdss_psf_at_points(hdu, *coords[0])


def download_sdss_frames(frame_list, bands, silence=False, n_threads=4):
    fits_to_stack = {k: [] for k in bands}
    with ThreadPool(n_threads) as p:
        keys = [
            (band, frame, silence)
            for band in bands
            for objid, frame in frame_list.iterrows()
        ]
        log.info('Downloading {} SDSS frames'.format(len(frame_list)))
        frame_data = p.starmap(__download_frame, keys)

    for (band, _, _), data in zip(keys, frame_data):
        if data is not None:
            fits_to_stack[band].append(
                get_data_from_fits(
                    data,
                    band
                )
            )
    return fits_to_stack


def extract_cutouts(input_frame_data, centre_pos, cutout_size):
    frames = pd.DataFrame(
        [],
        columns=('nelec', 'calib', 'sigma', 'sky', 'wcs',
                 'gain', 'darkvar')
    )
    for i, data in enumerate(input_frame_data):
        frame_wcs = data['frame_wcs']
        gain = data['gain']
        darkvar = data['darkvar']

        def make_cutout(data):
            return Cutout2D(
                data,
                centre_pos,
                cutout_size,
                wcs=frame_wcs,
                mode='partial',
                copy=True,
            )
        try:
            nelec_cutout = make_cutout(data['nelec'])
            calib_cutout = make_cutout(data['calib'])
            sky_cutout = make_cutout(data['sky'])
        except NoOverlapError:
            continue
        coverage_mask = np.isfinite(nelec_cutout.data)
        if np.any(coverage_mask):
            sigma = (
                calib_cutout.data
                * np.sqrt(nelec_cutout.data / gain**2 + darkvar)
            )
            coverage_mask[~coverage_mask] = np.nan
            frames.loc[i] = {
                # 'frame': frame,
                # 'image': img_cutout.data,
                'nelec': nelec_cutout.data,
                'calib': calib_cutout.data,
                'sigma': sigma,
                'sky': sky_cutout.data,
                'wcs': nelec_cutout.wcs,
                'gain': coverage_mask.astype(int) * gain,
                'darkvar': coverage_mask.astype(int) * darkvar,
            }
    return frames


def get_montage_cutout(frame_data, ra, dec, size, mosaic_directory='tmp__mosaic'):
    centre_pos = SkyCoord(ra, dec, unit=u.degree, frame='fk5')
    dx = 4 * float(size) * u.arcsec
    dy = 4 * float(size) * u.arcsec
    try:
        shutil.rmtree(mosaic_directory)
    except FileNotFoundError:
        pass
    with tempfile.TemporaryDirectory() as tmp_fits_dir:
        # write fits data to temporary files for montage
        [
            [
                fits.append(
                    os.path.join(tmp_fits_dir, 'out_{}.fits'.format(i)),
                    hdu.data,
                    hdu.header
                )
                for hdu in fd['fits']
            ]
            for i, fd in enumerate(frame_data['r'])
        ]
        # perform the mosaic
        montage.mosaic(tmp_fits_dir, mosaic_directory, background_match=True)

        # make a cutout of the desired region
        montage_wcs = WCS(os.path.join(mosaic_directory, 'mosaic.fits'))
        with fits.open(os.path.join(mosaic_directory, 'mosaic.fits')) as montage_result:
            cutout = Cutout2D(
                montage_result[0].data,
                centre_pos,
                (dx, dy),
                wcs=montage_wcs,
                mode='partial',
                copy=True,
            )

    return cutout


def stack_frames(frames):
    elec_stack = np.stack(frames['nelec'].values)
    calib_stack = np.stack(frames['calib'].values)
    sky_stack = np.stack(frames['sky'].values)
    g_stack = np.stack(frames['gain'].values)
    v_stack = np.stack(frames['darkvar'].values)
    pixel_count = np.isfinite(elec_stack).astype(int).sum(axis=0).astype(float)

    # or calculate from the raw data
    # I = C(n / g - S)
    I_combined = np.nansum(
        calib_stack * ((elec_stack / g_stack) - sky_stack),
        axis=0
    ) / pixel_count

    # sigma = 1/N * sqrt(sum(C**2 (n / g**2) + v))
    sigma_I = np.sqrt(
        np.nansum(
            calib_stack**2 * ((elec_stack / g_stack**2) + v_stack),
            axis=0
        )
    ) / pixel_count
    return I_combined, sigma_I


def sourceExtractImage(data, bkgArr=None, sortType='center', silence=False,
                       **kwargs):
    """Extract sources from data array and return enumerated objects sorted
    smallest to largest, and the segmentation map provided by source extractor
    """
    if not silence:
        log.info('Performing object detection using SourceExtractor')
    if bkgArr is None:
        bkgArr = np.zeros(data.shape)
    o = sep.extract(data.copy(), kwargs.pop('threshold', 0.05), segmentation_map=True,
                    **kwargs)
    if sortType == 'size':
        if not silence:
            log.info('Sorting extracted objects by radius from size')
        sizeSortedObjects = sorted(
            enumerate(o[0]), key=lambda src: src[1]['npix']
        )
        return sizeSortedObjects, o[1]
    elif sortType == 'center':
        if not silence:
            log.info('Sorting extracted objects by radius from center')
        centerSortedObjects = sorted(
            enumerate(o[0]),
            key=lambda src: (
                (src[1]['x'] - data.shape[0] / 2)**2
                + (src[1]['y'] - data.shape[1] / 2)**2
            )
        )[::-1]
        return centerSortedObjects, o[1]


def timer(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        dt = time.time() - t0
        log.info('Elapsed time: {:.2f}s'.format(dt))
        return res
    return wrapped_func


def get_rotation_correction(montaged_image, zoo_image, zoo_mask):
    target_im = np.ma.masked_array(zoo_image, zoo_mask)
    loss = np.inf
    for k in (0, 3):
        d = montaged_image / montaged_image.max() - np.rot90(target_im, k=k)
        loss_ = np.nansum(np.abs(d)) / d.size #+ np.sum(m)
        if loss_ < loss:
            rotation_correction = 2 * np.pi * k / 4
            loss = loss_
    return rotation_correction


@timer
def get_sdss_cutout(
    ra, dec,
    bands=['r'], cutout_radius=25, limit=1000,
    silence=False, return_frame_data=False
):
    """EXPLAIN EVERYTHING PLZ
    """
    centre_pos = SkyCoord(
        ra, dec,
        unit=u.degree, frame='fk5'
    )
    dx = (4 * float(cutout_radius) * u.arcsec) / (0.396 * u.arcsec)
    dy = 4 * float(cutout_radius) * u.arcsec / (0.396 * u.arcsec)
    cutout_size = (int(dx.value), int(dy.value))

    # Query SDSS for all nearby frames
    frame_list = get_frame_list(
        ra, dec,
        radius=2 * cutout_radius / 60,
        limit=limit,
        silence=silence
    )

    # download these frames and extract electron counts, calibration,
    # sky level, gain and darkvariance
    input_frame_data = download_sdss_frames(frame_list, bands, silence=silence)
    output = {}
    for band in bands:
        # cutout these images ready for stacking
        cutout_data = extract_cutouts(input_frame_data[band], centre_pos, cutout_size)

        # combine the frames into an image
        stacked_image, sigma_image = stack_frames(cutout_data)

        # ensures native byte order
        image_data = stacked_image.astype(np.float64)

        # source extract the image for masking
        objects, segmentation_map = sourceExtractImage(image_data)
        mask = np.logical_and(segmentation_map != objects[-1][0] + 1, segmentation_map != 0)
        mask[np.isnan(image_data)] = True

        psfs = []
        for (_, frame), data in zip(frame_list.iterrows(), input_frame_data['r']):
            psfs.append(get_frame_psf_at_points(frame, data['frame_wcs'], ra, dec))

        output[band] = {
            'data': np.ma.masked_array(stacked_image, mask),
            'sigma': np.ma.masked_array(sigma_image, mask),
            'psf': np.mean(psfs, axis=0),
            'wcs': cutout_data['wcs'].iloc[0]
        }
    if return_frame_data:
        return output, input_frame_data
    return output


if __name__ == '__main__':
    ra, dec = 135.033, 16.924
    cutout_radius = 25
    cutout_result = get_sdss_cutout(ra, dec, cutout_radius=cutout_radius, bands=['r'], silence=False)
    print(cutout_result['r']['data'].shape)
    print(cutout_result['r']['psf'].shape)
    import matplotlib.pyplot as plt
    from astropy.visualization import AsinhStretch
    s = AsinhStretch(0.3)
    plt.imshow(s(cutout_result['r']['data']), cmap='gray_r')
    plt.show()
