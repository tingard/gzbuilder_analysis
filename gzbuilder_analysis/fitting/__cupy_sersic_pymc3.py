import cupy as cp
import theano
import theano.tensor as tt
from cupyx.scipy.ndimage.filters import convolve as cuda_conv
from gzbuilder_analysis.rendering.cuda.sersic import sersic2d


def gen_grid(shape, oversample_n):
    x = cp.linspace(
        0.5 / oversample_n - 0.5,
        shape[1] - 0.5 - 0.5 / oversample_n,
        shape[1] * oversample_n
    )
    y = cp.linspace(
        0.5 / oversample_n - 0.5,
        shape[0] - 0.5 - 0.5 / oversample_n,
        shape[0] * oversample_n
    )
    return cp.meshgrid(x, y)


def downsample(render, oversample_n, size):
    return render.reshape(
        size[0], oversample_n, size[1], oversample_n
    ).mean(3).mean(1)


def convolve(render, psf, **kwargs):
    return cuda_conv(render, psf, mode='mirror')


def bulge_disk_render(
    cx, cy,
    mux=0, muy=0, Re=1, q=1, I=1, roll=0,
    bulge_dx=0, bulge_dy=0, bulge_scale=0.1, bulge_q=1, bulge_roll=0,
    bulge_frac=0.1, bulge_n=1
):
    if I == 0 or Re == 0:
        disk = cp.zeros(cx.shape)
        bulge = cp.zeros(cx.shape)
    else:
        #      sersic2d(x,  y,  mux, muy, roll, Re, q, c, I, n)
        disk = sersic2d(cx, cy, mux, muy, roll, Re, q, 2, I, 1)
        if bulge_scale == 0 or bulge_frac == 0:
            bulge = _p.zeros(cx.shape)
        else:
            disk_l = sersic_ltot(I, Re, 1)
            comp_l = disk_l * bulge_frac / (1 - bulge_frac)
            bulge_I = sersic_I(comp_l, bulge_scale * Re, bulge_n)
            bulge = sersic2d(
                cx, cy,
                mux + bulge_dx, muy + bulge_dy,
                bulge_roll, bulge_scale * Re,
                bulge_q, 2, bulge_I, bulge_n
            )
    return (disk + bulge)


class DiskBulgeBarOp(theano.Op):
    __props__ = ()
    itypes = [tt.dscalar] * 8
    otypes = [tt.dvector]

    def __init__(self, base_params, psf, spiral_distances, shape, oversample_n):
        self.base_params = base_params
        self.cx, self.cy = gen_grid(shape, oversample_n)
        self.oversample_n = oversample_n
        self.psf = psf


    def perform(self, node, inputs, owutput_storage):
        mux = cp.asarray(inputs[0])
        muy = cp.asarray(inputs[1])
        roll = cp.asarray(inputs[2])
        Re = cp.asarray(inputs[3])
        q = cp.asarray(inputs[4])
        I = cp.asarray(inputs[5])

        dx_bl = 0
        dx_bl = 0
        bt_bl = 0
        s_bl = 0
        q_bl = 1
        n_bl = 1

        dx_bl = 0
        dx_bl = 0
        bt_bl = 0
        s_bl = 0
        q_bl = 1
        n_bl = 1

        disk = sersic2d(self.cx, self.cy, mux, muy, roll, Re, q, c, I, n)
        bulge = sersic2d(self.cx, self.cy, mux + bulge_dx, muy, roll, Re, q, c, I, n)
        bar = sersic2d(self.cx, self.cy, mux, muy, roll, Re, q, c, I, n)
        spirals = cp.zeros(self.shape)
        render = convolve(
            downsample(
                disk + bulge + bar + spirals,
                self.oversample_n,
                size=self.shape
            ),
            self.psf
        )
        output_storage[0][0] = cp.asnumpy(render)

    def infer_shape(self, node, i0_shapes):
        return [Y.ravel().shape]
