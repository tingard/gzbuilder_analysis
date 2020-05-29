"""set up a flask server which accepts GET parameters of ra, dec,
radius and returns the stacked image, sigma image, pixel mask, PSF
and other metadata needed for fitting
"""
from flask import jsonify
from gzbuilder_analysis.data import get_sdss_cutout

# TODO: currently breaks after 1st request. Why?
# TODO: Make use of the NumpyJSONEncoder

def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """
    print('Recieved request')
    needed_args = {'ra', 'dec', 'size'}
    request_json = request.get_json(silent=True)
    request_args = request.args
    if request_json and needed_args.issubset(set(request_json)):
        params = request_json
    elif request_args and needed_args.issubset(set(request_args)):
        params = request_args
    else:
        return jsonify({'error': 'Invalid parameters'})

    ra = params['ra']
    dec = params['dec']
    size = params['size']
    print('-- parameters RA={:.4f}, DEC={:.4f}, SIZE={:.4f}'.format(
        ra, dec, size
    ))
    try:
        cutout_result = get_sdss_cutout(
            float(ra), float(dec),
            cutout_radius=float(size),
            bands=['r'],
            verbose=False,
        )
    except TypeError:
        return jsonify({'error': 'Invalid parameters'})

    converted_cutout_result = {}
    for band in cutout_result:
        mask = cutout_result[band]['data'].mask
        converted_cutout_result[band] = {
            k: v.tolist()
            for k, v in cutout_result[band].items()
        }
        converted_cutout_result[band]['mask'] = mask.tolist()

    print('--- Successfully created cutout result')

    return jsonify(converted_cutout_result)  # 'Hello {}!'.format(escape(name))
