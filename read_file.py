import io
import os
PATH = "./data/script"

for path, dirs, files in os.walk(PATH):
    for filename in files:
        fullpath = os.path.join(path, filename)
        print(filename)
        with io.open(fullpath, 'rb') as fn:
          lines = fn.readlines()
          for line in lines:
              print(line)
