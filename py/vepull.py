"""
kk:2017/09/03 run vs ve for mf
"""
 
from time import sleep
import sys, os, random, datetime, json
 
import string, re, shutil, math, subprocess , json, requests,arrow
  
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS
import oc, cv2
import numpy as np 
from StringIO import StringIO
 
  
def dlqklist(qklist, outdir = "veabc/"):
  print outdir
  nget = len(qklist)
  if not os.path.isdir(outdir):
    print outdir, "not found"
    return
  nn=0
  for qkreq in qklist:
    fullfn = outdir+"a"+str(qkreq)+".jpg"
    nn+=1
     
    if not os.path.isfile(fullfn):
      print 'getting',nn,"/",nget, fullfn
      oc.system("curl -s http://t0.tiles.virtualearth.net/tiles/a" + str(qkreq)+".jpeg?g=1398 > "+fullfn)
  
 
####################################################################
##    start...                                                             
####################################################################
  
 
 
 
if len(sys.argv) < 3:
  print "python", sys.argv[0], "aocgs212.txt 17 0.001"
  quit()
 
svgfile = "run15.svg"
#veroot = '/home/oc/F5T/oc/Flight-Imagery/work/ve/'
#scanroot = veroot
  
# print oc.qk2lonlat('1322322132033011222')
 
 
aocg = "lonlat.txt"  # input file
if len(sys.argv) > 1:
  aocg = sys.argv[1]
 
zoomget = 17
 
if len(sys.argv) > 2:
  zoomget = int(sys.argv[2])
 
 
 
print aocg
dist = 0.001 # half rect size in degrees to pull from
 
if len(sys.argv) > 3:
  dist = float(sys.argv[3])
 
 
 
 
 
 
outdir = "/home/oc/sdb1/ve/"
prevlon  = 0
prevlat = 0
nl = 0
 
pullfn = "tilelistdd.txt" # pull log
 
with open (pullfn, "w") as fh:
  fh.write("")
# 2015-04-02T10:17:39.411,-122.05697781,37.98234923182865,-17.5,0.0,0.0,2015-04-02T10:17:39.411,3.4398297742009163,2373.9941187882796,cameraoff
 
lonlats = [ ]
 
with open(aocg ) as fin:
  nline = 0
  for line in fin:
    nline +=1
  
    # print line
    row = line.strip().split(',')
    if len(row) < 2:
      continue
    lon =   float(row[0])
    lat =   float(row[1])
 
    # print abs(lon - prevlon) + abs(lat - prevlat)  
 
    if abs(lon - prevlon) + abs(lat - prevlat) < 0.01:
       continue
    lonlats.append([lon,lat])
  
#//lonlats = [ [-121.17, 37.70 ]]
#lonnlats = [[ -122.34, 38.38 ]]
print "lonlats", lonlats
 
for f in lonlats:
    print "lonlat ", f
 
    lon = f[0]
    lat = f[1]
 
    dlat = dist * math.cos(lat*3.141/180)
    poly = [[lon-dist,lat+dlat],[lon+dist,lat+dlat],[lon+dist,lat-dlat],[lon-dist,lat-dlat]]
 
    qklist = oc.corna2qklist([poly], zoomget)
    nget =  len(qklist)
    print nline, lon, lat, len(qklist)
    prevlon = lon
    prevlat = lat
    for gg in qklist:
      with open (pullfn, "a") as fh:
        fh.write(outdir+"a"+gg+".jpg\n")
 
    #print qklist   
    print "qklist len" , len(qklist) 
    dlqklist(qklist, outdir = outdir)
    nl+=1
    #if nl> 10:
    #  quit()
 
 
quit()