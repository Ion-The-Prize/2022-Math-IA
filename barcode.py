from graphics import *
import random
import numpy as np

def get_rand_color():
   r=random.randrange(25,256)
   b=random.randrange(25,256)
   g=random.randrange(25,256)
   color=color_rgb(r,b,g)
   return color

WHITE=color_rgb(255,255,255)
GREY=color_rgb(100,100,100)
YELLOW=color_rgb(255,255,0)
RED=color_rgb(255,0,0)

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

    title = None
    x_min = None  # type: float
    x_max = None  # type: float
    x_range = None  # type: float

    x_tick_marks=[] #type: list[float]


    y_min = None  # type: float
    y_max = None  # type: float
    y_range = None  # type: float

    win_width = None  # type: int
    win_height = None  # type: int
    pix_per_x = None  # type: int
    win = None  # type: GraphWin

    close_on_click = False
    # Bars so far
    bars = {}


    def __init__(self, title, _min_x, _max_x, _width=800, _height=300, num_colors=0, _num_ticks_x=10):
      """
      Initialize the barcode library

     :param _min_x: The value of x for the left edge of the barcode
     :param _max_x: The value of x for the right edge of the barcode
     :param _width: How wide the barcode should be
     :param _height: How tall the barcode should be
     :param num_colors: How many random colors will be requested. This can be 0
      """
      self.title = title
      self.x_min = _min_x
      self.x_max = _max_x
      self.num_ticks_x = _num_ticks_x

      x_range = self.x_max - self.x_min

      x_tick_first=None
      x_tick_inc=None
      if ( x_range>10 ):
          #integer mode
          x_tick_first= round(self.x_min)+1
          x_tick_inc = round(x_range/10)
      else:
          x_tick_first=self.x_min
          x_tick_inc=x_range/10
      for i in range(self.num_ticks_x):
          self.x_tick_marks.append(x_tick_first + i*x_tick_inc)

      self.win_width = _width
      self.win_height = _height
      self.win = GraphWin(str(title), self.win_width, self.win_height)

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
        assert(color_num is not None or color is not None)
        assert(color_num is None or color_num<len(self.colors))
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
        if not self.window_is_open():
            return

        if self.close_on_click and self.win.checkMouse():
            self.close()
            return

        self.win.autoflush=False
        rect = Rectangle(Point(0, 0), Point(self.win_width, self.win_height))
        rect.setFill('black');
        rect.draw(self.win)

        for x_pix in range(self.win_width):
            bar=self.bars.get(x_pix)
            if ( bar != None ):
                bar.draw()

        for tick_x in self.x_tick_marks:
            x_pix = (tick_x - self.x_min) * self.pix_per_x
            line = Line(Point(x_pix, 0), Point(x_pix, self.win_height))
            line.setFill(YELLOW)
            line.draw(self.win)
            label = Text(Point(x_pix-5, self.win_height-20), "{:.1f}".format(tick_x))
            label.setFill(RED);
            label.draw(self.win)

        self.win.flush()

    def window_is_open(self):
        return not self.win.closed

    def await_click(self):
        self.win.getMouse()

    def await_click_and_close(self):
        self.await_click()
        self.close()

    def close(self):
        self.win.close()

    def close_on_click(self):
        self.close_on_click = True
        
    def get_x_range(self):
        return np.arange(self.x_min, self.x_max, 1.0/self.pix_per_x)
