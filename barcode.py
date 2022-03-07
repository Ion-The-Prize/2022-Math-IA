import math

import graphics
from graphics import *
import random
import numpy as np
import webcolors

color_palette = list(webcolors.CSS21_NAMES_TO_HEX)
color_palette.remove('black')
color_palette.remove('gray')
color_palette.remove('silver')
color_palette.remove('white')

WHITE=color_rgb(255,255,255)
GREY=color_rgb(100,100,100)

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
        if self.y is None :
            y_top=0
        else:
            y_top = self.barcode.get_y_graph_pixel(self.y)

        line = Line(Point(self.x_pix, y_top), Point(self.x_pix, self.barcode.graph_height))
        line.setFill(self.color)
        line.draw(self.barcode.win)



class BarCode:
    # Colors always come from this web palette
    colors = color_palette
    item2color = dict()

    title = None
    x_min = None  # type: float
    x_max = None  # type: float
    x_range = None  # type: float

    x_tick_marks=[] #type: list[float]


    y_min = None  # type: float
    y_max = None  # type: float
    y_range = None  # type: float
    y_tick_marks=[] #type: list[float]

    win_width = None  # type: int
    graph_height = None  # type: int
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
      assert num_colors<len(self.colors)

      self.title = title
      self.x_min = _min_x
      self.x_max = _max_x
      self.num_ticks_x = _num_ticks_x

      self.x_tick_marks = self.get_tick_marks(self.x_min, self.x_max, 10)

      self.win_width = _width
      self.graph_height = _height
      self.win = GraphWin(str(title), self.win_width, self.graph_height+100)

      self.x_range = float(self.x_max-self.x_min)
      self.pix_per_x = float(self.win_width / self.x_range)

    def get_tick_marks(self, min, max, num_tick_marks):
        if min is None or max is None:
            return []

        range = max - min

        tick_first = None
        tick_inc = None
        if range > num_tick_marks:
            # integer mode
            tick_first = math.floor(min) + 1
            tick_inc = round(range / num_tick_marks)
        else:
            tick_first = min
            tick_inc = range / num_tick_marks

        tick = tick_first
        result=[]
        while tick < max:
            result.append(tick)
            tick += tick_inc
        return result

    def assign_color_number_to_item(self, item, color_number):
        self.item2color[item] = self.colors[color_number]


    def add_bar(self, x, color_num = None, color = None, color_item = None, y = None):
        """
        Draw a 'bar' (vertical line) on graph
        :param x: X value (must be between x_min and x_max)
        :param y: Y value (optional). Default is to draw line entire height of barcode.
        :param color_num: What randomly generated color to use.
        :param color_item: An "item" that was assigned a color via assign_color_number_to_item
        :param color: Color to draw
        :return:
        """
        assert(color_num is not None or color is not None or color_item is not None)
        assert(color_num is None or color_num<len(self.colors))
        assert(color_item is None or self.item2color[color_item] is not None)
        assert(self.x_min<=x<=self.x_max)

        if color_num:
            color=self.colors[color_num]

        if color_item:
            color=self.item2color[color_item]

        b=Bar(self, x, color, y)

        # Figure out implications of a Y value... do we need to rescale?
        recalculate=False
        if y != None:
            if self.y_min == None or y < self.y_min:
                #print("Found new y_min: {:.2f}".format(y))
                self.y_min = y
                recalculate = True
            if self.y_max == None or y > self.y_max :
                #print("Found new y_max: {:.2f}".format(y))
                self.y_max = y
                recalculate = True
            if recalculate :
                if self.y_min == self.y_max:
                    self.y_min = self.y_max - 1
                self.y_range = self.y_max - self.y_min

                self.y_tick_marks = self.get_tick_marks(self.y_min, self.y_max, 5)

                self.draw()

    def draw(self):
        if not self.window_is_open():
            return

        if self.close_on_click and self.win.checkMouse():
            self.close()
            return

        self.win.autoflush=False
        rect = Rectangle(Point(0, 0), Point(self.win_width, self.graph_height))
        rect.setFill('black')
        rect.draw(self.win)

        for x_pix in range(self.win_width):
            bar=self.bars.get(x_pix)
            if bar != None :
                bar.draw()

        for tick_x in self.x_tick_marks:
            x_pix = (tick_x - self.x_min) * self.pix_per_x
            line = Line(Point(x_pix, int(self.graph_height * 0.75)), Point(x_pix, self.graph_height))
            line.setFill(WHITE)
            line.draw(self.win)
            label = Text(Point(x_pix - 5, self.graph_height - 20), "{:.1f}".format(tick_x))
            label.setSize(18)
            label.setFill('white')
            label.draw(self.win)

        if len(self.y_tick_marks) > 0:
            for tick_y in self.y_tick_marks:
                y_pix = self.get_y_graph_pixel(tick_y)
                line = Line(Point(0, y_pix), Point(25, y_pix))
                line.setFill(WHITE)
                line.draw(self.win)
                label = Text(Point(20, y_pix), "{:.1f}".format(tick_y))
                label.setSize(18)
                label.setFill('white')
                label.draw(self.win)



        swatch_width=5
        swatch_height=30
        if len(self.item2color) == 0:
            # No items/values assigned to colors... just show colors along bottom
            for i in range(len(self.colors)):
                ul = Point(i*swatch_width,self.graph_height)
                lr = Point(ul.x+swatch_width, ul.y+swatch_height);
                box = Rectangle(ul, lr)
                box.setFill(self.colors[i])
                box.draw(self.win)
        else:
            # Color items are assigned, so try to line them up with X values
            # Because the can overlap, we adjust height of swatches
            swatch_vertical_step = swatch_height / 4
            swatch_vertical_start = self.graph_height - swatch_vertical_step
            i=0
            for item,color in self.item2color.items():
                swatch_vertical_start += swatch_vertical_step
                # Check to see if item is actually a number
                if type(item) == int or type(item) == float:
                    x_pix = (item - self.x_min) * self.pix_per_x

                    ul=Point(x_pix-swatch_width/2, swatch_vertical_start)
                    lr = Point(ul.x + swatch_width, ul.y + swatch_height);

                    box = Rectangle(ul, lr)
                    box.setFill(color)
                    box.draw(self.win)
                else:
                    # If the swatch item is not a number, then line swatch up along bottom
                    ul=Point(i*swatch_width, self.graph_height+2*swatch_height)
                    lr = Point(ul.x + swatch_width, ul.y + swatch_height);

                    box = Rectangle(ul, lr)
                    box.setFill(self.colors[i])
                    box.draw(self.win)
                i += 1

        self.win.flush()

    def get_y_graph_pixel(self, y):
        y_fraction = (y - self.y_min) / self.y_range

        y_pix= self.graph_height * (1-y_fraction)
        return y_pix


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
