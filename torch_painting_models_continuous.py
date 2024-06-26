import torch
from torch import nn
import torchgeometry
import torchvision.transforms as T
import warnings
import numpy as np
import os
import array as arr
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')

from options import Options 
from continuous_brush_model import StrokeParametersToImage, special_sigmoid
import tps_warp
from PIL import Image
opt = Options()
opt.gather_options()

stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
h, w = stroke_shape[1], stroke_shape[1]
# print('stroke_shape', stroke_shape)
if h != opt.max_height:
    w = int(opt.max_height/h*w)
    h = opt.max_height
#

# Make the canvas shape consistent with the content shape
content = Image.open(opt.objective_data[0])
content = np.array(content)
content_h, content_w = content.shape[0], content.shape[1]
w = int((opt.max_height/content_h)*content_w)
h = int(opt.max_height)
print("New w h:", w, h)
##

h_og, w_og = h, w

def get_param2img(h_full, w_full, n_stroke_models=1):
    # Crop
    hs, he = int((0.5-opt.stroke_curva/2)*h_full), int((0.5+opt.stroke_curva/2)*h_full)
    ws, we = int((0.5-opt.stroke_length/2)*w_full), int((0.5-opt.stroke_length/2)*w_full)

    cropped_h = int(opt.stroke_curva*h_full)
    cropped_w = int(opt.stroke_length*w_full)

    param2imgs = []
    pad_for_full = []

    param2img = StrokeParametersToImage(cropped_h, cropped_w)
    try:
        param2img.load_state_dict(torch.load(
            os.path.join(opt.cache_dir, 'param2img.pt')))
    except RuntimeError as e:
        print('Ignoring "' + str(e) + '"')
    param2img.eval()
    param2img.to(device)
    param2imgs.append(param2img)

    pad = T.Pad((ws, hs, w_full-cropped_w-ws, h_full-cropped_h-hs))
    pad_for_full.append(pad)

    return param2imgs, pad_for_full

param2imgs, pad_for_full = get_param2img(h_og, w_og)

cos = torch.cos
sin = torch.sin
def rigid_body_transform(a, xt, yt, anchor_x, anchor_y):
    # a is the angle in radians, xt and yt are translation terms of pixels
    # anchor points are where to rotate around (usually the center of the image)
    # Blessed be Peter Schorn for the anchor point transform https://stackoverflow.com/a/71405577
    A = torch.zeros(1, 3, 3).to(device)
    a = -1.*a
    A[0,0,0] = cos(a)
    A[0,0,1] = -sin(a)
    A[0,0,2] = anchor_x - anchor_x * cos(a) + anchor_y * sin(a) + xt#xt
    A[0,1,0] = sin(a)
    A[0,1,1] = cos(a)
    A[0,1,2] = anchor_y - anchor_x * sin(a) - anchor_y * cos(a) + yt#yt
    A[0,2,0] = 0
    A[0,2,1] = 0
    A[0,2,2] = 1
    return A

class RigidBodyTransformation(nn.Module):
    def __init__(self, a, xt, yt):
        super(RigidBodyTransformation, self).__init__()

        self.xt = nn.Parameter(torch.ones(1)*xt)
        self.yt = nn.Parameter(torch.ones(1)*yt)
        self.a = nn.Parameter(torch.ones(1)*a)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        anchor_x, anchor_y = w/2, h/2

        # M = rigid_body_transform(self.weights[0], self.weights[1]*(w/2), self.weights[2]*(h/2), anchor_x, anchor_y)
        M = rigid_body_transform(self.a[0], self.xt[0]*(w/2), self.yt[0]*(h/2), anchor_x, anchor_y)
        with warnings.catch_warnings(): # suppress annoing torchgeometry warning
            warnings.simplefilter("ignore")
            return torchgeometry.warp_perspective(x, M, dsize=(h,w))

class BrushStroke(nn.Module):
    def __init__(self, 
                stroke_length=None, stroke_z=None, stroke_bend=None,
                color=None, 
                a=None, xt=None, yt=None):
        super(BrushStroke, self).__init__()

        # if color is None: color=torch.rand(3).to(device)
        # if color is None: color=(torch.rand(3).to(device)/10)+0.45
        if color is None: color=(torch.rand(3).to(device)*.4)+0.3
        if a is None: a=(torch.rand(1)*2-1)*3.14
        if xt is None: xt=(torch.rand(1)*2-1)
        if yt is None: yt=(torch.rand(1)*2-1)


        if stroke_length is None: stroke_length=torch.rand(1)*(.05-.01) + .01
        if stroke_z is None: stroke_z = torch.rand(1)
        if stroke_bend is None: stroke_bend = torch.rand(1)*.04 - .02 
        stroke_bend = min(stroke_bend, stroke_length) if stroke_bend > 0 else max(stroke_bend, -1*stroke_length)

        self.transformation = RigidBodyTransformation(a, xt, yt)
        
        self.stroke_length = stroke_length

        self.stroke_z = stroke_z
        self.stroke_bend = stroke_bend
        #self.stroke_bend = 0

        self.stroke_length.requires_grad = True
        self.stroke_z.requires_grad = True
        self.stroke_bend.requires_grad = True

        self.stroke_length = nn.Parameter(self.stroke_length)
        self.stroke_z = nn.Parameter(self.stroke_z)
        self.stroke_bend = nn.Parameter(self.stroke_bend)

        self.color_transform = nn.Parameter(color)

    def forward(self, h, w):
        # Do rigid body transformation
        full_param = torch.zeros((1,12)).to(device)

        #
        #print("Stroke length:", self.stroke_length)
        #print("Stroke z:", self.stroke_z)
        
        # X
        full_param[0,0] = 0
        full_param[0,3] = self.stroke_length/3 
        full_param[0,6] = 2*self.stroke_length/3
        full_param[0,9] = self.stroke_length
        #full_param[0,3] = 0.15/3 
        #full_param[0,6] = 2*0.15/3
        #full_param[0,9] = 0.15
        # Y
        full_param[0,1] = 0
        full_param[0,4] = self.stroke_bend
        full_param[0,7] = self.stroke_bend
        #full_param[0,4] = 0
        #full_param[0,7] = 0
        full_param[0,10] = 0
        # Z
        full_param[0,2] = 0.2
        #full_param[0,2] = self.stroke_z
        full_param[0,5] = self.stroke_z
        full_param[0,8] = self.stroke_z
        full_param[0,11] = 0.2
        #full_param[0,11] = self.stroke_z

        model_ind = np.random.randint(len(param2imgs))
        stroke = param2imgs[model_ind](full_param).unsqueeze(0)

        squeezed_stroke = np.squeeze(stroke)
        if type(squeezed_stroke) is torch.Tensor:
            squeezed_stroke = squeezed_stroke.detach().cpu().numpy()
        exp_sqz_stroke = np.expand_dims(squeezed_stroke, axis=2)

        from plan import show_img, show_img2, img_for_tps

        stroke = pad_for_full[model_ind](stroke)

        x = self.transformation(stroke)
        x = special_sigmoid(x)

        # show_img(x)
        x = torch.cat([x,x,x,x], dim=1)
        # print('forawrd brush', x.shape)
        # Color change
        x = torch.cat((x[:,:3]*0 + self.color_transform[None,:,None,None], x[:,3:]), dim=1)
    
        return x

    def make_valid(stroke):
        with torch.no_grad():
            og_len = stroke.stroke_length.item()
            stroke.stroke_length.data.clamp_(0.01,0.05)
            
            stroke.stroke_bend.data.clamp_(-1*stroke.stroke_length, stroke.stroke_length)
            stroke.stroke_bend.data.clamp_(-.02,.02)

            stroke.stroke_z.data.clamp_(0.1,1.0)

            # stroke.transformation.weights[1:3].data.clamp_(-1.,1.)
            stroke.transformation.xt.data.clamp_(-1.,1.)
            stroke.transformation.yt.data.clamp_(-1.,1.)


            stroke.color_transform.data.clamp_(0.02,0.98)

class Painting(nn.Module):
    def __init__(self, n_strokes, background_img=None, brush_strokes=None):
        # h, w are canvas height and width in pixels
        super(Painting, self).__init__()
        self.n_strokes = n_strokes

        self.background_img = background_img

        if self.background_img.shape[1] == 3: # add alpha channel
            t =  torch.zeros((1,1,self.background_img.shape[2],self.background_img.shape[3])).to(device)
            # t[:,:3] = self.background_img
            self.background_img = torch.cat((self.background_img, t), dim=1)

        if brush_strokes is None:
            self.brush_strokes = nn.ModuleList([BrushStroke() for _ in range(n_strokes)])
        else:
            self.brush_strokes = nn.ModuleList(brush_strokes)

    def get_optimizers(self, multiplier=1.0):
        xt = []
        yt = []
        a = []
        length = []
        z = []
        bend = []
        color = []
        
        for n, p in self.named_parameters():
            if "xt" in n.split('.')[-1]: xt.append(p)
            if "yt" in n.split('.')[-1]: yt.append(p)
            if "a" in n.split('.')[-1]: a.append(p)
            if "stroke_length" in n.split('.')[-1]: length.append(p)
            if "stroke_z" in n.split('.')[-1]: z.append(p)
            if "stroke_bend" in n.split('.')[-1]: bend.append(p)
            if "color_transform" in n.split('.')[-1]: color.append(p)

        position_opt = torch.optim.RMSprop(xt + yt, lr=5e-3*multiplier)
        rotation_opt = torch.optim.RMSprop(a, lr=1e-2*multiplier)
        color_opt = torch.optim.RMSprop(color, lr=5e-3*multiplier)
        bend_opt = torch.optim.RMSprop(bend, lr=3e-3*multiplier)
        length_opt = torch.optim.RMSprop(length, lr=1e-2*multiplier)
        thickness_opt = torch.optim.RMSprop(z, lr=1e-2*multiplier)

        return position_opt, rotation_opt, color_opt, bend_opt, length_opt, thickness_opt


    def forward(self, h, w, use_alpha=True, strokes=None):
        if self.background_img is None:
            canvas = torch.ones((1,4,h,w)).to(device)
        else:
            canvas = T.Resize(size=(h,w))(self.background_img).detach()

        canvas[:,3] = 1 # alpha channel

        mostly_opaque = False#True

        for brush_stroke in self.brush_strokes:
            single_stroke = brush_stroke(h,w)

            if mostly_opaque: single_stroke[:,3][single_stroke[:,3] > 0.5] = 1.
            
            if use_alpha:
                canvas = canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
            else:
                canvas = canvas[:,:3] * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke[:,:3]
        return canvas


    def to_csv(self):
        ''' To csv string '''
        csv = ''
        for bs in self.brush_strokes:
            # Translation in proportions from top left
            # x = str((bs.transformation.weights[1].detach().cpu().item()+1)/2)
            # y = str((bs.transformation.weights[2].detach().cpu().item()+1)/2)
            # r = str(bs.transformation.weights[0].detach().cpu().item())
            x = str((bs.transformation.xt[0].detach().cpu().item()+1)/2)
            y = str((bs.transformation.yt[0].detach().cpu().item()+1)/2)
            r = str(bs.transformation.a[0].detach().cpu().item())
            length = str(bs.stroke_length.detach().cpu().item())
            thickness = str(bs.stroke_z.detach().cpu().item())
            bend = str(bs.stroke_bend.detach().cpu().item())
            color = bs.color_transform.detach().cpu().numpy()
            csv += ','.join([x,y,r,length,thickness,bend,str(color[0]),str(color[1]),str(color[2])])
            csv += '\n'
        csv = csv[:-1] # remove training newline
        return csv

    def validate(self):
        ''' Make sure all brush strokes have valid parameters '''
        for s in self.brush_strokes:
            BrushStroke.make_valid(s)


    def cluster_colors(self, n_colors):
        colors = [b.color_transform[:3].detach().cpu().numpy() for b in self.brush_strokes]
        colors = np.stack(colors)[None,:,:]

        from sklearn.cluster import KMeans
        from paint_utils import rgb2lab, lab2rgb
        
        # Cluster in LAB space
        colors = rgb2lab(colors)
        kmeans = KMeans(n_clusters=n_colors, random_state=0)
        kmeans.fit(colors.reshape((colors.shape[0]*colors.shape[1],3)))
        colors = [kmeans.cluster_centers_[i] for i in range(len(kmeans.cluster_centers_))]

        colors = np.array(colors)

        # Back to rgb
        colors = lab2rgb(colors[None,:,:])[0]
        return torch.from_numpy(colors).to(device)# *255., labels