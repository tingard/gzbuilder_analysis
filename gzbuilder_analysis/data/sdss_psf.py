import sys
# from numpy import *
import numpy as np
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter
from .fitsUtils import fits_table, table_fields

# Returns (a, sigma1, b, sigma2)
def sdss_dg_psf_params(psfield, band):
    T = fits_table(psfield[6].data)
    # the psf table has one row.
    assert(len(T)) == 1
    T = T[0]
    # http://www.sdss.org/dr7/dm/flatFiles/psField.html
    # good = PSP_FIELD_OK
    if T.status[band] != 0:
        print('T.status[band=%s] =' % band, T.status[band])
    assert(T.status[band] == 0)
    a = 1.0
    s1 = T.psf_sigma1_2g[band]
    b = T.psf_b_2g[band]
    s2 = T.psf_sigma2_2g[band]
    return (float(a), float(s1), float(b), float(s2))


def sdss_dg_psf_apply(img, dgparams):
    (a, s1, b, s2) = dgparams
    a /= (s1**2 + b*s2**2)
    return a * (s1**2 * gaussian_filter(img, s1)
                + b * s2**2 * gaussian_filter(img, s2))


def sdss_dg_psf(dgparams, shape=(51, 51)):
    img = np.zeros(shape)
    h, w = shape
    img[h/2, w/2] = 1.
    return sdss_dg_psf_apply(img, dgparams)


# Reconstruct the SDSS model PSF from KL basis functions.
#   hdu: the psField hdu for the band you are looking at.
#      eg, for r-band:
#         psfield = pyfits.open('psField-%06i-%i-%04i.fit' % (run,camcol,field))
#        bandnum = 'ugriz'.index('r')
#         hdu = psfield[bandnum+1]
#
#   x,y can be scalars or 1-d numpy arrays.
# Return value:
#    if x,y are scalars: a PSF image
#    if x,y are arrays:  a list of PSF images
def sdss_psf_at_points(hdu, x, y):
    rtnscalar = np.isscalar(x) and np.isscalar(y)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    psf = table_fields(hdu.data)

    psfimgs = None
    (outh, outw) = (None, None)

    # From the IDL docs:
    # http://photo.astro.princeton.edu/photoop_doc.html#SDSS_PSF_RECON
    #   acoeff_k = SUM_i{ SUM_j{ (0.001*ROWC)^i * (0.001*COLC)^j * C_k_ij } }
    #   psfimage = SUM_k{ acoeff_k * RROWS_k }
    for k in range(len(psf)):
        nrb = psf.nrow_b[k]
        ncb = psf.ncol_b[k]

        c = psf.c[k].reshape(5, 5)
        c = c[:nrb, :ncb]

        (gridi, gridj) = np.meshgrid(range(nrb), range(ncb))

        if psfimgs is None:
            psfimgs = [np.zeros_like(psf.rrows[k]) for xy in np.broadcast(x, y)]
            (outh, outw) = (psf.rnrow[k], psf.rncol[k])
        else:
            assert(psf.rnrow[k] == outh)
            assert(psf.rncol[k] == outw)

        for i, (xi, yi) in enumerate(np.broadcast(x,y)):
            acoeff_k = np.sum(((0.001 * xi)**gridi * (0.001 * yi)**gridj * c))
            if False:  # DEBUG
                print('coeffs:', (0.001 * xi)**gridi * (0.001 * yi)**gridj)
                print('c:', c)
                for (coi, ci) in zip(((0.001 * xi)**gridi * (0.001 * yi)**gridj).ravel(), c.ravel()):
                    print('co %g, c %g' % (coi, ci))
                print('acoeff_k', acoeff_k)
            psfimgs[i] += acoeff_k * psf.rrows[k]

    psfimgs = [img.reshape((outh, outw)) for img in psfimgs]

    if rtnscalar:
        return psfimgs[0]
    return psfimgs


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run, camcol, field = sys.argv[1:4]
        psfield = fits.open('psField-%06i-%i-%04i.fit' % (run, camcol, field))
        bandnum = 'ugriz'.index('r')
        hdu = psfield[bandnum+1]
    elif len(sys.argv) == 2:
        psfield = fits.open(sys.argv[1])
        bandnum = 'ugriz'.index('r')
        hdu = psfield[bandnum+1]
    import matplotlib.pyplot as plt
    psf = sdss_psf_at_points(hdu, 10, 10)
    plt.imshow(psf[15:35, 15:35])
    plt.colorbar()
    plt.show()
