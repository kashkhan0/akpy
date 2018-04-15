from time import sleep
import sys, os, random, datetime, json

import string, re
from decimal import Decimal
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS
import linecache
import math
import subprocess 
import oc 
import json 
import requests
import arrow
import argparse
from time import sleep, time
import cv2
import numpy as np
 
import shutil 
import sys
import os, shutil
import PIL
from PIL import Image
from os import walk
from os import listdir
from os.path import isfile, join
 
from PIL.ExifTags import TAGS
 
# def quit():
#   drawmap()
#   oc.svgwrite('ocg'+Config.date+'.svg',  w=Config.svgw, h=Config.svgh)  
#   sys.exit()

def drawmap():
  pass

def system(cmd, debug = False):
  if debug:
    print cmd
  return os.system(cmd)

def pnames(path):
    fsplit=path.strip().split('/')
    lenf = len(fsplit)
    sname = fsplit[lenf-4]+"_"+fsplit[lenf-3]+"_"+fsplit[lenf-2]+"_"+fsplit[lenf-1]
    data = sname.replace(".JPG","")

    out = {"path":path, 'data':data, "iname": fsplit[-1] , "inum":  int(path[-8:-4]), 'd3200':fsplit[-2], 'date' : fsplit[-4], "im15":data+"/s"+ fsplit[-1] }
    return out

def genpath(imnum, base):
    d = imnum%1000
    d32 = (imnum - d)/1000
    out =  base + '1'+'%02d'%d32 + 'D3200/DSC_'+'%04d'%d+'.JPG'
    if d == 0 and d32 == 0: out =  base + '100D3200/DSC_0001.JPG'
    if d == 0 and d32 > 0: 
        out =  base + '1'+'%02d'%d32 + 'D3200/DSC_0001.JPG'
    return out

def findprev(seektime, timelist):
  key = seektime
  for n in xrange(10):
    key= seektime-n

    if key in timelist:
      return key
  return -1  
      
def findnext(seektime, timelist):
  key = seektime
  for n in xrange(10):
    key= seektime+n

    if key in timelist:
      return key
  return -1    

def  getpos(seektime, timelist):
  try:
    ll1= timelist[seektime]
  except:
    kp =findprev(seektime, timelist)
    kn = findnext(seektime,timelist)
    if kp < 0 or kn < 0:
      return -1
    llprev= pts[ kp]
    llnext= pts[kn]
    print llprev
    print llnext
    lat=(llprev['lat']+llnext['lat'])/2
    lon=(llprev['lon']+llnext['lon'])/2
    ll1={"lon":lon, "lat": lat}

      
  return ll1

alist = ['aocgs20171117.txt']
fdbfn = 'outfdb20171117swlist.txt'
# alist = ['aocgs20151025.txt']
# alist = ['aocgs20150815.txt']
# alist = ['aocgs20151019.txt']

# alist = ['aocgs20150815.txt']


# alist = ['aocgs20150104.txt']
# alist = ['aocgs20140513.txt']


astart = {}

# astart['20151025']={"imnum":10, "lon": -121.87052249908447,"lat": 37.44174589921417  }

 
for afile in alist:
  aftrim =  afile[5:-4]
  singlefn = aftrim+'single.txt'
  
  print 'singlefn', singlefn
  print 'fdbfn', fdbfn
  if not os.path.isfile(singlefn) :
    print 'no single', singlefn
    continue  
  if not os.path.isfile(fdbfn) :
    print 'no fdb', fdbfn
    continue
  
  with  open(singlefn, "r") as mf:
    singlelines = mf.readline()
  r=singlelines.strip().split(',')
  try:
    lon1 = float(r[1])
    lat1 = float(r[2])
    imurisingle = r[0]

  except:
    print 'badfile', singlefn
    continue

  print 'imurisingle',imurisingle ,lon1, lat1

  
  imfdblist={}    

  with  open(fdbfn, "r") as mf:
    fdblines = mf.readlines()

  print  len(fdblines),'lines in',fdbfn  
 
  for item in fdblines:

    r=item.strip().split(',')
    if len(r) < 2: continue
    imtime = r[1]
    imurifdb = r[0]
    
    comp = imurifdb.split('/')
    imuri ="/"+comp[-4]+"/"+comp[-3]+"/"+comp[-2]+ "/"+comp[-1]
    #print imuri
    # if  comp[2] != "wf00":
    #   continue
    # if imurifdb !=  '/20151121/wf00/100D3200/DSC_0050.JPG':
    #   continue
    # print comp[2], imurifdb
    # quit()
    try:tsf = float(r[6])
    except: tsf = -1
    imfdblist[imuri]={"imtime": imtime, 'imuri':imuri, 'tsf': tsf}  


  print imurisingle, lon1 , lat1
  print imurisingle 
  print 'imfdblist',imfdblist[imurisingle]
 

  seektime = arrow.get(imfdblist[imurisingle]['imtime'], 'YYYY:MM:DD HH:mm:ss').timestamp
  # print 'seektime', seektime
 
  with open(afile , 'r') as aocg:
    aocglines = aocg.readlines()


  prevDistBetween = 0
  fileLineNumForLoc  = 0 
  lfix = ''

  pts={}
  stime=0
  etime=0

  nt=0
  #find start 
  distBetween = -1
  for tline in aocglines:

    row=tline.strip().split(',')
    if len(row) < 2:
        continue

    lonCurr = float(row[1])
    latCurr = float(row[2])
    tfix = (row[0])
 
    # print 'curr', lonCurr, latCurr
    distBetween = oc.calcdist((latCurr, lonCurr), (lat1, lon1))
    print "distBetween", distBetween
    
    #if distBetween > prevDistBetween and prevDistBetween < 0.8 and prevDistBetween != 0:
    if latCurr > lat1:
      lfix = tline
      try:
        ocgtime = arrow.get(tfix ).timestamp
      except:
        ocgtime = arrow.get(tfix[:-3] ).timestamp
      print "distBetween", distBetween

      break

    fileLineNumForLoc += 1
    prevDistBetween = distBetween

     

  print afile, str(fileLineNumForLoc),lonCurr, latCurr, distBetween, prevDistBetween
  print 'timestamp ocg', ocgtime
 


  diff =  seektime - ocgtime
  print "checkdiff", seektime,'-', diff ,'=' , ocgtime, '=', seektime-diff

  with open( 'pts.txt', "w") as myfile:
    myfile.write('')


  mintime = ocgtime
  ntime = 0

  for tline in aocglines:
    row=tline.strip().split(',')
    if len(row) < 2:
        continue

    lonCurr = float(row[1])
    latCurr = float(row[2])
    tfix = (row[0])
    try:
      utc = arrow.get(row[0] )
    except:
      print   'row[0]', row[0]
      utc = arrow.get(row[0][:-3] )
    curtime = utc.timestamp
    try:
      avuyvkbk= tsprev
    except:
      tsprev = curtime-1      
    tdiff = curtime-tsprev
   
 
    tsprev = curtime
 

    try:  
      pts[curtime]={'lon': float(row[1]), 'lat':float(row[2])}
      mintime = min(mintime, curtime)
      
      with open( 'pts.txt', "a") as myfile:
        myfile.write(str(curtime)+': { "lon": '+str(row[1])+', "lat":'+str(row[2])+'}\n')
    except:
      print 'time problem', curtime
      quit()
    # if ntime >50:
    #   quit()    
    ntime+=1  
  print 'mintime', mintime
  out = ''  
  with open( 'centers'+aftrim+'.txt', "w") as myfile:
     myfile.write(out)
  with open( 'cdebug'+aftrim+'.txt', "w") as myfile:
     myfile.write(out)   
  tprev=0
  ngood = 0
  nbad = 0
  nim=0
  last =""
  print "last", sorted(fdblines)[-1]
  for item in sorted(fdblines):
    nim+=1
    r=item.strip().split(',')
   # print item
    try:
      comp = r[0].split("/")
      imuri ="/"+comp[-4]+"/"+comp[-3]+"/"+comp[-2]+ "/"+comp[-1]
   
      # print 'imlines im', im
    except:
      print 'nogood',im
      quit()
      continue  
 
    # print imfdblist[imuri], 

    imagetime = arrow.get(imfdblist[imuri]['imtime'], 'YYYY:MM:DD HH:mm:ss').timestamp
    seektime = int(imagetime-diff)
    with open( 'cdebug'+aftrim+'.txt', "a") as myfile:
        myfile.write(imuri + ' ' +str(imagetime)+ ' ' + str(seektime)  + '\n') 
    if seektime<mintime:
      continue
    ll1 = getpos( seektime, pts)
    try:
      ll1['lon']
    except:  
      print 'nbad', seektime , imuri
      print 'imagetime', imagetime
      nbad +=1    
      continue
    tprev=stime
    utc=arrow.utcnow()
    time= float(utc.timestamp) + float(utc.microsecond)/1000
    o = imuri+','+ str(ll1['lon'])+','+ str(ll1['lat'])+'\n'

#    last = o
    with open( 'centers'+aftrim+'.txt', "a") as myfile:
       myfile.write(o)
    if ngood%100==0:
      print o



#  print last
  print 'ngood', ngood, nbad   


os.system("python centers2jssw20171118.py " +  'centers'+aftrim+'.txt')


quit()








