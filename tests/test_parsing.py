import json
import numpy as np
import pandas as pd
import gzbuilder_analysis.parsing as pg


TEST_ANNOTATION = [
    {
        'task': 'disk',
        'task_label': None,
        'value': [
            {
                'task': 'drawDisk',
                'task_label': 'Draw an ellipse around the outer edge of the galaxy',
                'value': [{
                    'x': 100,
                    'y': 101,
                    'rx': 20,
                    'ry': 15,
                    'tool': 0,
                    'angle': 40,
                    'frame': 0,
                    'tool_label': 'Galactic Disk '
                }]
            },
            {'task': 'scaleSlider', 'value': 1.6},
            {'task': 'T13', 'value': 0.1}
        ]
    },
    {
        'task': 'bulge',
        'task_label': None,
        'value': [
            {
                'task': 'drawBulge',
                'task_label': 'If the galaxy has a bulge, draw it!',
                'value': [{
                    'x': 102,
                    'y': 103,
                    'rx': 15,
                    'ry': 15,
                    'tool': 0,
                    'angle': 41,
                    'frame': 0,
                    'tool_label': 'Galactic Bulge '
                }]
            },
            {'task': 'scaleSlider', 'value': 1.7},
            {'task': 'intensitySlider', 'value': 0.4},
            {'task': 'sersicSlider', 'value': '1'}
        ]
    },
    {
        'task': 'bar',
        'task_label': None,
        'value': [
            {
                'task': 'drawBar',
                'task_label': 'Draw on a galaxy bar if you see one',
                'value': [{
                    'x': 104,
                    'y': 105,
                    'tool': 0,
                    'angle': 42,
                    'frame': 0,
                    'width': 90,
                    'height': 20,
                    'tool_label': 'Galactic Bar '
                }]
            },
            {'task': 'scaleSlider', 'value': '1'},
            {'task': 'T16', 'value': 0.3},
            {'task': 'T15', 'value': 0.6},
            {'task': 'shapeSlider', 'value': 2.2}
        ]
    },
    {
        'task': 'spiral',
        'task_label': None,
        'value': [
            {
                'task': 'drawSpiral',
                'task_label': 'Draw any spiral arms you see (requires you to have drawn a disc)',
                'value': [
                    {
                        'tool': 0,
                        'frame': 1,
                        'points': [
                            {'x': 0, 'y': 0},
                            {'x': 1, 'y': 1},
                            {'x': 2, 'y': 2},
                            {'x': 3, 'y': 3},
                            {'x': 4, 'y': 4},
                        ],
                        'details': [{'value': 0.18}, {'value': '1'}],
                        'tool_label': 'Spiral Arm'
                    }
                ]
            },
            {'task': 'falloffSlider', 'value': '1'}
        ]
    }
]

TEST_CLASSIFICATION = pd.Series(dict(
    annotations=json.dumps(TEST_ANNOTATION),
    user_id=1234,
    user_name='test_user',

))

TEST_IMAGE_SIZE = (200, 200)


# This model should be the result of parsing the above classification
TEST_MODEL = pd.Series({
    ('disk', 'mux'): 100,
    ('disk', 'muy'): 99,
    ('disk', 'q'): 15 / 20,
    ('disk', 'Re'): 20 * 2 / 3 * 1.6,
    ('disk', 'roll'): np.deg2rad(130.0),
    ('disk', 'I'): 0.0625,
    ('bulge', 'mux'): 102,
    ('bulge', 'muy'): 97,
    ('bulge', 'q'): 1,
    ('bulge', 'Re'): 8.5,
    ('bulge', 'roll'): np.deg2rad(41),
    ('bulge', 'I'): 0.25,
    ('bulge', 'n'): 1.0,
    ('bar', 'mux'): 149,
    ('bar', 'muy'): 85,
    ('bar', 'q'): 2/9,
    ('bar', 'Re'): 30.0,
    ('bar', 'roll'): np.deg2rad(48),
    ('bar', 'I'): 0.1875,
    ('bar', 'n'): 0.6,
    ('bar', 'c'): 2.2,
})


def assert_model_is_correct(model):
    ks = TEST_MODEL.keys()
    is_wrong_at = 'Incorrect parameters:' + (
        ', '.join([
            ' '.join(i)
            for i in np.array(ks)[~np.isclose(
                [TEST_MODEL[k] for k in ks],
                [model[k[0]][k[1]] for k in ks]
            )]
        ])
    )
    is_correct = np.allclose(
        [TEST_MODEL[k] for k in ks],
        [model[k[0]][k[1]] for k in ks]
    )
    assert is_correct, is_wrong_at


def test_parse_annotation():
    try:
        model = pg.parse_annotation(
            TEST_ANNOTATION,
            image_size=TEST_IMAGE_SIZE
        )
    except (KeyError, IndexError):
        assert False, "Exception while parsing annotation"
    assert_model_is_correct(model), "Parsed model is not correct"


def test_parse_classification():
    try:
        model = pg.parse_classification(
            TEST_CLASSIFICATION,
            image_size=TEST_IMAGE_SIZE
        )
    except (KeyError, IndexError):
        assert False, "Exception while parsing classification"
    assert_model_is_correct(model)


"""
Still need to add tests for
- scaling
- reprojection
- reparametrization
"""
