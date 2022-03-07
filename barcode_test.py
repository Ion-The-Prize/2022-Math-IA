import barcode, random

barcode=barcode.BarCode(-10, 15, 800, 200, 10)

i=0
while True:
  i += 1
  barcode.add_bar(-10 + random.random()*25, random.randrange(10), y=random.random() * 500)
  if ( i % 500 == 0):
    barcode.draw()
