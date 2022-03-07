from graphics import *
import random

def get_rand_color():
   r=random.randrange(256)
   b=random.randrange(256)
   g=random.randrange(256)
   color=color_rgb(r,b,g)
   return color

WHITE=color_rgb(255,255,255)

class Bar:
    """A bar on the barcode"""
    barcode=None
    x=None
    x_pix = None
    y=None
    color=None

    def __init__(self, barcode, x, color, y=None, ):
        self.barcode=barcode
        self.x=x
        self.x_pix = int((x-barcode.x_min) * barcode.pix_per_x)

        self.y=y
        self.color=color

        barcode.bars[self.x_pix] = self

    def draw(self):
        if ( self.y == None ):
            y_top = 0
        else:
            y_top=1.0*self.barcode.win_height*(self.y-self.barcode.y_min)/self.barcode.y_range;

        line = Line(Point(self.x_pix, y_top), Point(self.x_pix, self.barcode.win_height))
        line.setFill(self.color)
        line.draw(self.barcode.win)


class BarCode:
    colors = []

    x_min = None  # type: float
    x_max = None  # type: float
    x_range = None  # type: float

    y_min = None  # type: float
    y_max = None  # type: float
    y_range = None  # type: float

    win_width = None  # type: int
    win_height = None  # type: int
    pix_per_x = None  # type: int
    win = None  # type: GraphWin

    # Bars so far
    bars = {}


    def __init__(self, _min_x, _max_x, _width=800, _height=300, num_colors=0):
      """
      Initialize the barcode library

     :param _min_x: The value of x for the left edge of the barcode
     :param _max_x: The value of x for the right edge of the barcode
     :param _width: How wide the barcode should be
     :param _height: How tall the barcode should be
     :param num_colors: How many random colors will be requested. This can be 0
      """
      self.x_min = _min_x
      self.x_max = _max_x
      self.win_width = _width
      self.win_height = _height
      self.win = GraphWin("Barcode", self.win_width, self.win_height)

      self.x_range = float(self.x_max-self.x_min)
      self.pix_per_x = float(self.win_width / self.x_range)

      for i in range(num_colors):
        self.colors.append(get_rand_color())

    def add_bar(self, x, color_num = None, color = None, y = None):
        """
        Draw a 'bar' (vertical line) on graph
        :param x: X value (must be between x_min and x_max)
        :param y: Y value (optional). Default is to draw line entire height of barcode.
        :param color_num: What randomly generated color to use.
        :param color: Color to draw
        :return:
        """
        assert(color_num != None or color != None)
        assert(color_num == None or color_num<len(self.colors))
        assert(self.x_min<=x<=self.x_max)

        if (color_num):
            color=self.colors[color_num]

        b=Bar(self, x, color, y)

        # Figure out implications of a Y value... do we need to rescale?
        recalculate=False
        if ( y != None ):
            if ( self.y_min == None or y < self.y_min ):
                self.y_min = y
                recalculate = True
            if ( self.y_max == None or y > self.y_max ):
                self.y_max = y
                recalculate = True
            if ( recalculate ):
                self.y_range = self.y_max - self.y_min
                if ( self.y_range == 0 ):
                    # Avoid division-by-zero errors later
                    self.y_range = 1

    def draw(self):
        self.win.autoflush=False
        rect = Rectangle(Point(0, 0), Point(self.win_width, self.win_height))
        rect.setFill('black');
        rect.draw(self.win)

        for x_pix in range(self.win_width):
            bar=self.bars.get(x_pix)
            if ( bar != None ):
                bar.draw()
        self.win.flush()