from graphics import *
import random

def get_rand_color():
   r=random.randrange(256)
   b=random.randrange(256)
   g=random.randrange(256)
   color=color_rgb(r,b,g)
   return color

colors=[]

x_min=None # type: float
x_max=None # type: float
x_range=None # type: float

y_min=None # type: float
y_max=None # type: float
y_range=None # type: float

win_width=None  # type: int
win_height=None # type: int
pix_per_x=None #type: int
win=None #type: GraphWin

# Bars so far
bars={}

class Bar:
    """A bar on the barcode"""
    x=None
    x_pix = None
    y=None
    color=None

    def __init__(self, x, color, y=None, ):
        self.x=x
        self.x_pix = int((x-x_min) * pix_per_x)

        self.y=y
        self.color=color

        bars[self.x_pix] = self

    def draw(self):
        if ( self.y == None ):
            y_top = 0
        else:
            y_top=1.0*win_height*(self.y-y_min)/y_range;

        line = Line(Point(self.x_pix, y_top), Point(self.x_pix, win_height))
        line.setFill(self.color)
        line.draw(win)


def init(_min_x, _max_x, _width=800, _height=300, num_colors=0):
  """
  Initialize the barcode library

 :param _min_x: The value of x for the left edge of the barcode
 :param _max_x: The value of x for the right edge of the barcode
 :param _width: How wide the barcode should be
 :param _height: How tall the barcode should be
 :param num_colors: How many random colors will be requested. This can be 0
  """
  global x_min, x_max, win_width, win_height, x_range, pix_per_x, win
  x_min = _min_x
  x_max = _max_x
  win_width = _width
  win_height = _height
  win = GraphWin("Barcode", win_width, win_height)

  x_range = 1.0*x_max-x_min
  pix_per_x = 1.0 * win_width / x_range

  global colors
  for i in range(num_colors):
    colors.append(get_rand_color())

def add_bar(x, color_num = None, color = None, y = None):
    """
    Draw a 'bar' (vertical line) on graph
    :param x: X value (must be between x_min and x_max)
    :param y: Y value (optional). Default is to draw line entire height of barcode.
    :param color_num: What randomly generated color to use.
    :param color: Color to draw
    :return:
    """
    assert(color_num != None or color != None)
    assert(color_num == None or color_num<len(colors))
    assert(x_min<=x<=x_max)

    if (color_num):
        color=colors[color_num]

    b=Bar(x,color, y)

    # Figure out implications of a Y value... do we need to rescale?
    global y_min, y_max, y_range
    recalculate=False
    if ( y != None ):
        if ( y_min == None or y < y_min ):
            y_min = y
            recalculate = True
        if ( y_max == None or y > y_max ):
            y_max = y
            recalculate = True
        if ( recalculate ):
            y_range = y_max - y_min
            if ( y_range == 0 ):
                # Avoid division-by-zero errors later
                y_range = 1

def draw():
    win.autoflush=False
    rect = Rectangle(Point(0, 0), Point(win_width, win_height))
    rect.setFill('black');
    rect.draw(win)

    for x_pix in range(win_width):
        bar=bars.get(x_pix)
        if ( bar != None ):
            bar.draw()
    win.flush()