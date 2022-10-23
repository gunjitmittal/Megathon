import os
import shutil
directory = "train"
Lang = []

def dirlen(dir):
      i=0
      for file in os.listdir(dir):
            filename = os.fsdecode(file)
            if filename == ".DS_Store":
              continue
            i+=1
      return i

for lang in os.listdir(os.fsencode('train')):
    langname = os.fsdecode(lang)
    if langname == ".DS_Store":
     continue
    Lang.append(langname[0:])
print(Lang)
for l in Lang:
    if dirlen((os.path.join(os.fsencode("train"),os.fsencode(l))))==1:
        for file in os.listdir(os.path.join(os.fsencode("train"),os.fsencode(l))):
              if (os.fsdecode(file) != ".DS_Store"):
                filename = os.fsdecode(os.path.join(os.fsencode("train"),os.fsencode(l),file))
                shutil.copy(filename, os.path.join(os.fsencode("train"),os.fsencode(l),os.fsencode("Copy.mp3")))