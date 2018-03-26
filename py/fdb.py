import sys
import os, shutil
import PIL
from PIL import Image
from os import walk
from os import listdir
from os.path import isfile, join
import os.path, time, math
import arrow
from PIL.ExifTags import TAGS


def system(cmd, debug = False):
  if debug:
    print cmd
  return os.system(cmd)

def process(cmd, debug = False):
  if debug:
    print cmd
  res=subprocess.check_output(cmd, shell=True)
  utc = arrow.utcnow()
  local = utc.to('US/Pacific')

# def get_field (exif,field) :
#   for (k,v) in exif.iteritems():
#      if TAGS.get(k) == field:
#         return v


utc = arrow.utcnow().format('YYYY-MM-DD HH:mm:ss ZZ')
args= ' '.join(sys.argv)
with open("pylog.txt", "a") as myfile:
        myfile.write(utc + " python "+ args +"\n")

inlistfn = 'dsclist20170312-33.txt'
outfn = "fdb20170312-33.txt"
flist = {}
root = 'F08/FI/'
with open(inlistfn,'r') as infile:
  for line in infile:
    path = line.strip()[7:]
    print path
    flist[path] = path

with open(outfn, "w") as imlog:
  imlog.write( "")

for fname in sorted(flist):

    nd=0
    nf=0
    utcp = arrow.utcnow()
   

     
    nd+=1
   
    nf+=1 
    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(root+fname)
    tt=arrow.get(mtime).format('YYYY-MM-DD HH:mm:ss ZZ')
    try:
      img = PIL.Image.open(root+fname)
      print fname
      exif = img._getexif()
    except:
      continue

    DateTimeOriginal="1900"
    FocalLength=(10,10)
    ExposureTime=(10,10)
    FNumber=(10,10)
    ISOSpeedRatings=0
    
    for tag, value in exif.items():
      decoded = TAGS.get(tag, tag)
      # if decoded!="MakerNote": print "decoded", decoded, value
      if decoded=="DateTimeOriginal": DateTimeOriginal=value
      if decoded=="FocalLength": FocalLength=value
      if decoded=="ExposureTime": ExposureTime=value
      if decoded=="FNumber": FNumber=value
      if decoded=="ISOSpeedRatings": ISOSpeedRatings=value
      if decoded=="SubsecTimeOriginal":SubsecTimeOriginal=value
      # FocalLength
      # ExposureTime
      # MaxApertureValue
      # ret[decoded] = value
    # print  dirName+"/"+fname, float(FNumber[0])/float(FNumber[1]),(ExposureTime[1])/(ExposureTime[0])
    fullpath = fname 
    sp = fullpath.split('/')
    imuri = fname

    ts1= arrow.get(DateTimeOriginal, 'YYYY:MM:DD HH:mm:ss').timestamp
    ts1= float(SubsecTimeOriginal)/100+ts1

    try: 
      td = ts1-tsp
      td2= math.floor(ts1)-math.floor(tsp)
    except: 
      td = -1.0 
      td2= -1 
    tsp=ts1
    exdata= imuri+"," +DateTimeOriginal+"," +str(FocalLength[0]/FocalLength[1])+"," +str(float(FNumber[0])/float(FNumber[1])) + ","+str((ExposureTime[1])/(ExposureTime[0]))+","+str(ISOSpeedRatings)+","+"%0.1f"%(ts1)+","+"%0.1f"%(td)+","+"%0d"%(td2)



    if (nf%100) == 0: 
      utc = arrow.utcnow()

      print nf,utc.format('HH:mm:ss'),  exdata

    with open(outfn, "a") as imlog:
      imlog.write(exdata +"\n")
      


print "nf",nf




