import barcode, random

barcode1=barcode.BarCode("Testing", -10, 15, 800, 200, 10)
barcode2=barcode.BarCode("Testing2", -10, 15, 800, 200, 10)

barcode1.close_on_click()

barcode1.assign_color_number_to_item(-10, 0)
barcode1.assign_color_number_to_item(5, 1)
i=0
while barcode1.window_is_open():
  i += 1
  barcode1.add_bar(-10 + random.random() * 25, random.randrange(10), y=random.random() * 500)
  if ( i % 500 == 0):
    barcode1.draw()
