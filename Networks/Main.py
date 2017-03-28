import numpy
import scipy.ndimage as nd
import scipy
import caffe
import PIL.Image
from pylab import *
from cStringIO import StringIO
from IPython.display import clear_output, Image, display
from google.protobuf import text_format



# PATH
MODEL_PATH            = '../Models/'
NET_FN     = MODEL_PATH + 'googlenet.deploy.prototxt'

PRETRAINED_MODEL_PATH = '../Pretrained/'
PARAM_FN   = PRETRAINED_MODEL_PATH + 'bvlc_googlenet.caffemodel'

DATA_PATH  = '../Data/'

# GLOBAL PARAMETERS
Model = None
Net   = None

def ShowArray(a, fmt = 'jpeg'):
    figure(1)
    scipy.misc.imsave("testimg.jpg", a)
    image = PIL.Image.open("testimg.jpg")
    imshow(image)
    show()

def LoadModel():
    global Model, Net

    caffe.set_device(0)

    # Load model
    if Model is None:
        Model = caffe.io.caffe_pb2.NetParameter()
        text_format.Merge(open(NET_FN).read(), Model)
        Model.force_backward = True
        open('tmp.prototxt', 'w').write(str(Model))

        Net = caffe.Classifier('tmp.prototxt', PARAM_FN,
                               mean = numpy.float32([104.0, 116.0, 122.0]),
                               channel_swap = (2, 1, 0))

def Preprocess(net, img):
    return numpy.float32(numpy.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def Deprocess(net, img):
    return numpy.dstack((img + net.transformer.mean['data'])[::-1])

def ObjectiveL2(dst):
    dst.diff[:] = dst.data

def MakeStep(net,
             stepSize = 1.5,
             end = 'inception_4c/output',
             jitter = 32,
             clip = True,
             objective = ObjectiveL2):
    ''' Basic gradient ascent step '''
    src = net.blobs['data']
    dst = net.blobs[end]

    ox, oy = numpy.random.randint(-jitter, jitter + 1, 2)
    src.data[0] = numpy.roll(numpy.roll(src.data[0], ox, -1), oy, -2)

    net.forward(end=end)
    objective(dst)
    net.backward(start=end)
    g = src.diff[0]

    src.data[:] += stepSize / numpy.abs(g).mean() * g
    src.data[0] = numpy.roll(numpy.roll(src.data[0], -ox, -1), -oy, -2)

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = numpy.clip(src.data, -bias, 255 - bias)

def DeepDream(net,
              baseImg,
              iterN = 10,
              octaveN = 4,
              octaveScale = 1.4,
              end = 'inception_4c/output',
              clip = True,
              **stepParams):
    # Prepare base image for all actaves
    octaves = [Preprocess(net, baseImg)]
    for i in range(octaveN - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octaveScale, 1.0 / octaveScale), order = 1))

    src = net.blobs['data']
    detail = numpy.zeros_like(octaves[-1])
    for octave, octaveBase in enumerate(octaves[::-1]):
        h, w = octaveBase.shape[-2:]
        if octave > 0:
            # Upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order = 1)

        src.reshape(1,3,h,w)
        src.data[0] = octaveBase + detail
        for i in range(iterN):
            MakeStep(net, end = end, clip = clip, ** stepParams)

            # Visualization
            vis = Deprocess(net, src.data[0])
            if not clip:
                vis = vis * (255.0 / numpy.percentile(vis, 99.98))
            ShowArray(vis)
            print octave, i, end, vis.shape
            clear_output(wait = True)

        # Extract details produced on the current octave
        detail = src.data[0] - octaveBase
    return Deprocess(net, src.data[0])

def Test():
    global Net
    img = numpy.float32(PIL.Image.open(DATA_PATH + 'sky1024px.jpg'))
    ShowArray(img)
    _ = DeepDream(Net, img)

if __name__ == '__main__':
    LoadModel()
    Test()