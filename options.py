import argparse 

# Based off of Jun-Yan Zhu's Cyclegan implementation's options

class Options(object):
    def __init__(self):
        self.initialized = False
        self.opt = {}

        self.INIT_TABLE_Z = 0.1

        # Dimensions of canvas in meters
        # CANVAS_WIDTH  = 0.3047 # 12"
        # CANVAS_HEIGHT = 0.2285 # 9"
        # self.CANVAS_WIDTH  = 0.254 -0.005# 10"
        # self.CANVAS_HEIGHT = 0.2032 -0.005# 8"
        self.CANVAS_WIDTH  = 0.3556 -0.005# 14"
        self.CANVAS_HEIGHT = 0.2794 -0.005# 11"


        self.CANVAS_WIDTH_PIX  = None # set these after taking a picture
        self.CANVAS_HEIGHT_PIX = None

        # X,Y of canvas wrt to robot center (global coordinates)
        # self.CANVAS_POSITION = (0,.5) 
        self.CANVAS_POSITION = (0, .5-.04)

        """ How many times in a row can you paint with the same color before needing more paint """
        self.GET_PAINT_FREQ = 3

        self.HOVER_FACTOR = 0.1


        # Number of cells to paint in x and y directions
        self.cells_x, self.cells_y = 4, 5

        # Dimensions of the cells in Meters
        #cell_dim = (0.0254, 0.0508) #h/w in meters. 1"x2"
        self.cell_dim_y, self.cell_dim_x = self.CANVAS_HEIGHT / self.cells_y, self.CANVAS_WIDTH / self.cells_x

        # The brush stroke starts halfway down and 20% over from left edge of cell
        self.down = 0.5 * self.cell_dim_y
        self.over = 0.2 * self.cell_dim_x

    def initialize(self, parser):
        parser.add_argument('--simulate', action='store_false')
        parser.add_argument('--use_cache', action='store_false')
        parser.add_argument('--dont_plan', action='store_true', help='Use saved plan from last run')

        parser.add_argument('--stroke_length', default=0.4, type=float)
        parser.add_argument('--stroke_curva', default=0.2, type=float)


        parser.add_argument('--max_height', default=512, type=int, help='How much to downscale canvas')
        parser.add_argument('--n_colors', default=30, type=int, help='Number of colors of paint to use')
        parser.add_argument("--cache_dir", type=str,
            default='cache', help='Where to store cached files.')
        parser.add_argument("--tensorboard_dir", type=str,
            default='./log', help='Where to write tensorboard log to.')
        parser.add_argument("--global_it", type=int,
            default=0, help='Picking up where it left off.')


        parser.add_argument('--num_strokes', type=int, default=500)
        parser.add_argument('--n_stroke_models', type=int, default=1)

        parser.add_argument('--adaptive', action='store_true')
        parser.add_argument('--generate_whole_plan', action='store_true')
        parser.add_argument('--strokes_before_adapting', type=int, default=100)
        parser.add_argument('--remove_prop', type=float, default=0.8, help="Proportion of strokes to remove when adapting")

        parser.add_argument('--adapt_optim_iter', type=int, default=30)



        parser.add_argument('--objective', nargs='*', type=str, help='clip_conv_loss|l2')
        parser.add_argument('--objective_data', nargs='*', type=str)
        parser.add_argument('--objective_weight', nargs='*', type=float, default=1.0)
        parser.add_argument('--optim_iter', type=int, default=400)
        parser.add_argument('--lr_multiplier', type=float, default=0.4)
        parser.add_argument('--init_lr', type=float, default=3e-2, help="learning rate for initial objective")

        parser.add_argument('--init_objective', nargs='*', type=str, help='clip_conv_loss|l2')
        parser.add_argument('--init_objective_data', nargs='*', type=str)
        parser.add_argument('--init_objective_weight', nargs='*', type=float, default=1.0)
        parser.add_argument('--init_optim_iter', type=int, default=40)
        parser.add_argument('--n_inits', type=int, default=0, help='Number of times to try different initializations')

        parser.add_argument('--intermediate_optim_iter', type=int, default=40)
        parser.add_argument('--use_colors_from', type=str, default=None, help="Get the colors from this image. \
                None if you want the colors to come from the optimized painting.")

        parser.add_argument('--num_augs', type=int, default=30)
        parser.add_argument('--bin_size', type=int, default=3000)

        parser.add_argument('--plan_gif_dir', type=str, default='cache')
        parser.add_argument('--log_frequency', type=int, default=5)
        parser.add_argument('--middle_result_name', type=str, default='mid_result.jpg')
        parser.add_argument('--base_canvas', type=str, default='default.png')

        return parser 

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(description="Sawyer Painter")
            parser = self.initialize(parser)

        self.opt = vars(parser.parse_args())


    def __getattr__(self, attr_name):
        return self.opt[attr_name]