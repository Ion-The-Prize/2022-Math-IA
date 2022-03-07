import barcode, random

barcode=barcode.BarCode("Testing", -10, 15, 800, 200, 10)
barcode.close_on_click()

i=0
while barcode.window_is_open():
  i += 1
  barcode.add_bar(-10 + random.random()*25, random.randrange(10), y=random.random() * 500)
  if ( i % 500 == 0):
    barcode.draw()
