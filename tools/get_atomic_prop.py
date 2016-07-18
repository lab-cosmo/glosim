import csv
from collections import defaultdict

pname='Electro-Negativity'
#pname='Atomic Weight'
columns = defaultdict(list)
with open('elements.csv') as csvfile:
   reader = csv.DictReader(csvfile)
   for row in reader:
       for (k,v) in row.items():
          try :
             v=int(v)
          except:
             try:
               v=float(v)
             except:
               continue
          columns[k].append(v)

pdict=dict(zip(columns['Atomic Number'],columns[pname]))
print pdict
