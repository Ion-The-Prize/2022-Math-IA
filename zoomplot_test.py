from polynomials import *

zoom_poly = Polynomial([20901888000.0, -3449433600.0, -5168875680.0, 265617576.0, 324864000.0, -5646078.0, -7647696.0, 32562.0, 61236.0],[-5 - 5/6, 6 + 6/7, -3.0, -6.0, 7 + 1/9, 5.0, 2.0, -6 - 2/3])
zoom_poly.open_barcode_window(-15,15,1600)
plot = ZoomPlot(zoom_poly, color_points_with_newton_root=True)
plt.show()

input("Press Enter to continue...")