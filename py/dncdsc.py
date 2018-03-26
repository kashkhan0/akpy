

"""
kk:2015/10/29 g2 for locating 
"""

from time import sleep
import sys, os, random, datetime, json

import string, re
from decimal import Decimal
from PIL import Image, ImageFont, ImageDraw
from PIL.ExifTags import TAGS
import linecache
import math
import subprocess 
import   oc
import json 
import requests
import arrow
import argparse
from time import sleep, time
import cv2
import numpy as np
import shutil
from oc import Config 
import sqlite3
import copy
import struct
import base64,zlib
from array import array
from StringIO import StringIO

class Imgobj(object):
    pass



 

def system(cmd, debug = False):
  if debug:
    print cmd
  return os.system(cmd)

def process(cmd, debug = False, logName="proclog.txt"):
  if debug:
    print cmd
  res=subprocess.check_output(cmd, shell=True)
  utc = arrow.utcnow()
  local = utc.to('US/Pacific')
  if logName != None:
    with open(logName, "a") as myfile:
      myfile.write( "\n\n" + str(local) + "\n" +  cmd + "\n" + res)  
  return res

def detectCompute(img1, akazeThreshold=0.001):
  """
  """

  detector = cv2.AKAZE_create()
  detector.setThreshold(akazeThreshold)
  kp1, desc1 = detector.detectAndCompute(img1, None)
  return kp1, desc1 



def tnow( text = "tt"):
  e = arrow.utcnow()
  return  e.timestamp + e.microsecond / 1e6


def genpath(imnum=1234, base="/home/oc/F5T/oc/Flight-Imagery/2015-09-19/DCIM000720151019/"):
  """
  inputs
    imnum: image number in sequence
    base: sdcard base directory
  return:
    pathname after checking if file exists 
    or -1 if it doesn't 
  """  
  d = imnum%1000
  d32 = (imnum - d)/1000
  out =  base + '1'+'%02d'%d32 + 'D3200/DSC_'+'%04d'%d+'.JPG'
  if d == 0 and d32 == 0: out =  base + '100D3200/DSC_0001.JPG'
  if d == 0 and d32 > 0: 
      out =  base + '1'+'%02d'%d32 + 'D3200/DSC_0001.JPG'

  if not os.path.isfile(out):
    return -1    
  return out

def pnames(path):
  """
  input: image path
  output: object with converted paths and names
  """
  fsplit=path.strip().split('/')
  lenf = len(fsplit)
  sname = fsplit[lenf-4]+"_"+fsplit[lenf-3]+"_"+fsplit[lenf-2]+"_"+fsplit[lenf-1]
  data = sname.replace(".JPG","/")
  seqnum = 1000 * int(fsplit[-2][1:3]) + int(path[-8:-4])
  base = path[:-21]
  out = {"path":path, 'base': base, 'data':data, "iname": fsplit[-1] , "inum":  seqnum,'d3200':fsplit[-2], 'date' : fsplit[-4], "im15":data+"/s"+ fsplit[-1] }
  return out



def dummycopy(src,dst):
    Config.fcount+=1
    shutil.copy2(src,dst)
    if Config.fcount%100==0:
        utc=arrow.utcnow()
        newms=utc.timestamp + utc.microsecond/1e6
 
        print str(newms - Config.ms)+":", Config.fcount , Config.fcount /(newms - Config.ms), "per sec"
    return    


def copytreek(src, dst, symlinks=False, ignore=None, copy_function=dummycopy,
             ignore_dangling_symlinks=False):
    """Recursively copy a directory tree.

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied. If the file pointed by the symlink doesn't
    exist, an exception will be added in the list of errors raised in
    an Error exception at the end of the copy process.

    You can set the optional ignore_dangling_symlinks flag to true if you
    want to silence this exception. Notice that this has no effect on
    platforms that don't support os.symlink.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    The optional copy_function argument is a callable that will be used
    to copy each file. It will be called with the source path and the
    destination path as arguments. By default, copy2() is used, but any
    function that supports the same signature (like copy()) can be used.

    """
    names = os.listdir(src)

    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    os.makedirs(dst)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.islink(srcname):
                linkto = os.readlink(srcname)
                if symlinks:
                    os.symlink(linkto, dstname)
                else:
                    # ignore dangling symlink if the flag is on
                    if not os.path.exists(linkto) and ignore_dangling_symlinks:
                        continue
                    # otherwise let the copy occurs. copy2 will raise an error
                    copy_function(srcname, dstname)
            elif os.path.isdir(srcname):
                copytreek(srcname, dstname, symlinks, ignore, copy_function)
            else:
                # Will raise a SpecialFileError for unsupported file types
                copy_function(srcname, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        # except Error as err:
        #     errors.extend(err.args[0])
        except EnvironmentError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        shutil.copystat(src, dst)
    except OSError as why:
        if WindowsError is not None and isinstance(why, WindowsError):
            # Copying file access times may fail on Windows
            pass
        else:
            errors.append((src, dst, str(why)))
    if errors:
        raise Error(errors)



def doNothing(a):
  if a[-12:-9] != 'DSC':
    return
  # print "/"+a[len(Config.FI):]
  # posix.stat_result(st_mode=33252, st_ino=94374552, st_dev=2097, st_nlink=1, st_uid=1000, st_gid=1000, st_size=13674390, st_atime=1444216024, st_mtime=1391470672, st_ctime=1441306724)
  stat = os.stat(a)
  
  str2 = "Flight-Imagery";

 
  imuri = a[a.find(str2)+14:]
  # nexist = getcount(imuri)
  # # print imuri

  # if nexist>0:
  #   # print imuri, 'exists'
  #   return
 
  utc=arrow.utcnow()
  time= float(utc.timestamp) + float(utc.microsecond)/1000
  data={   
  'imuri': imuri,
  'st_size': stat.st_size,
  'st_mtime': stat.st_mtime,
  'md5': '',
  'n': 3333333,
  't': time, 
  'time':time,
  "stage":"dcim"
  }
  # print data


  check={   
  'imuri': imuri,
  "stage":"dcim"
  }



  count = Config.collection.count(check)
  # print count
  if count ==0 :
    Config.collection.insert(data)
  # appendqueue(data)
  # sys.exit()
 
  return



def scantreek(src,  symlinks=False, ignore=None, scan_function=doNothing,
             ignore_dangling_symlinks=False):
    """Recursively scan a directory tree.

 

    """
    names = os.listdir(src)
    # os.makedirs(dst)
    errors = []
    for name in sorted(names):
        Config.count+=1
        if  Config.count%100==0:
          print Config.count
 
        srcname = os.path.join(src, name)
        # dstname = os.path.join(dst, name)
        try:
            if os.path.islink(srcname):
                linkto = os.readlink(srcname)
                if symlinks:
                    os.symlink(linkto, dstname)
                else:
                    # ignore dangling symlink if the flag is on
                    if not os.path.exists(linkto) and ignore_dangling_symlinks:
                        continue
                    # otherwise let the copy occurs. copy2 will raise an error
                    scan_function(srcname)
            elif os.path.isdir(srcname):
                scantreek(srcname)
            else:
                # Will raise a SpecialFileError for unsupported file types
                scan_function(srcname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        # except Error as err:
        #     errors.extend(err.args[0])
        except EnvironmentError as why:
            errors.append((srcname, dstname, str(why)))

    if errors:
        raise Error(errors)







def ingestDSC():
  """
  imports images from DCIM
 
    1. check if sdcards inserted
    2. check if new images on card
    3. copy images from card to first repo
    4. compute and store md5 in copystatus in db
    5. read and store exif = stat in db
    6. mirror images onsite and offsite
    7. clear sdcard once more than 2 copies exist

  """

  return "success"




def scanDSC(path):
  """
  imports images from DCIM
 
    1. check if sdcards inserted
    2. check if new images on card
    3. copy images from card to first repo
    4. compute and store md5 in copystatus in db
    5. read and store exif = stat in db
    6. mirror images onsite and offsite
    7. clear sdcard once more than 2 copies exist

  """
  

  for a in path:
    pathtoscan= Config.FI +a
    scantreek(pathtoscan)
 
  return "success"




def dncDSC2DBold(fi,processed  ,AKAZEThreshold=0.001, maxn=0, testdnc=0):
  """
  - check for new DSC in db
  - resize image
  - detect and compute AKAZE features
  - store in db in px coords
  - update DSC DNC flag in db
  """
  n=0
  while True:

    r = requests.get("http://10.0.0.4/F5T/oc/Flight-Imagery/work/pylink/repo/dncdsc.php?count="+str(Config.count))  
    Config.count +=1


    n+=1

    # r = requests.post("http://10.0.0.4/F5T/oc/Flight-Imagery/work/pylink/control.php", data = { "stage":"dncDSC2DB", "last" : Config.last })
    

    # print "r.text",str(Config.count)+"/"+str(maxn),r.text 
 
 
    # if  r.text==''  or r.text=='die'  or r.text=='null' or r.text=='none' : 
    #   print 'none found'
    #   print r.url
    #   break
    try:
      js=json.loads(r.text )
      
    except:
      print 'bad json'
      print r.url
      break
    if len(js)  <1:
      print 'no more images'
      break
    
    for sd in range(len(js)):
       
      imuri=js[sd]['imuri']

    
      rfi =   requests.get("http://10.0.0.4/F5T/oc/Flight-Imagery/work/pylink/repo/getstatus.php?&stage=dncDSC2DBdone&imuri="+imuri)

      if rfi.text!='null': 
        print 'exists',n, imuri 
        continue
   

      print "count", Config.count,imuri,

      rfi =   requests.get("http://10.0.0.4/F5T/oc/Flight-Imagery/work/pylink/repo/geturl.php?imuri="+imuri)
 
      fi = rfi.text
      inpath = fi + imuri
      processed_path =  processed + imuri +'/'
      res = process('mkdir -p '+ processed_path)
      
     
      detector = cv2.AKAZE_create()
      detector.setThreshold(AKAZEThreshold)

      try: 
        r = requests.get(inpath)
        im = Image.open(StringIO(r.content))
        (width, height) = im.size
        swidth, sheight = width*0.16,height*0.16
        im.thumbnail( (swidth, sheight ), Image.ANTIALIAS)
      except:  
        print 'imge dl prob', inpath
        quit()
        # r = requests.post("http://10.0.0.4/F5T/oc/Flight-Imagery/work/pylink/control.php", data = { "stage":"dncDSC2DBdone", "imuri" : imuri, 'kp': 0, "timetaken" :  0 , 'AKAZEThreshold':AKAZEThreshold  , "status": "resizefail"})
        continue

      img1 = np.array(im) 
      img1 = img1[:, :, ::-1].copy() 
      # img1 = cv2.imread(inpath)
      # small = cv2.resize(img1, (0,0), fx=0.16, fy=0.16) 
      
      cv2.imwrite(processed_path+"small.jpg", img1)
      cv2.imwrite("small.jpg", img1)
      ts=tnow() 
      kp1, desc1 = detector.detectAndCompute(img1, None)

      te=tnow()
      timetaken = te-ts
      print len(kp1), "kp", te-ts , "s"
      # ['__class__', '__delattr__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'angle', 'class_id', 'octave', 'pt', 'response', 'size']
      feats=''

      n1=0
      for nkp in range(len(kp1)):
        # print 'kp',len(kp1),(kp1[0].__str__),len(desc1[0])
        kp = kp1[nkp]
        float_array = [kp.pt[0], kp.pt[1], kp.size,kp.angle,  kp.response, kp.octave]
        keyp = struct.pack('6f', kp.pt[0], kp.pt[1], kp.size, kp.angle,  kp.response, kp.octave)
        # if (n1==0): print float_array
        g=keyp+desc1[nkp].tostring()
     
        # desc1[nkp].tostring()
        ss=np.fromstring(g[-61:], dtype=np.uint8)
        feats += g
        n1+=1
        # feats+= float_array.tostring() +  desc1[nkp].tostring()

     
      # featz=(base64.b64encode(feats))
      with open(processed_path+'feats.bin', 'wb') as f:
        f.write(feats)
      # print processed_path+'feats.bin', 'written'  
      r = requests.get("http://10.0.0.4/F5T/oc/Flight-Imagery/work/pylink/repo/dncdsc.php?stage=dncDSC2DBdone&imuri="+imuri+"&kp="+str(len(kp1))+"&timetaken="+str(timetaken)+"&width="+str(swidth)+"&height="+str(sheight)+"&status=good&AKAZEThreshold=0.001" )
      # print r.url

      # print 'rtext', r.text

 

   


def dncDSC2DB(imuri, fi,processed  ,AKAZEThreshold=0.0005, maxn=0, testdnc=0):
  """
  - resize image
  - detect and compute AKAZE features
  - store in db in px coords
  - update DSC DNC flag in db
  """
 
  inpath = fi + imuri
  processed_path =  processed + imuri +'/'
  if os.path.isfile(processed_path+"small.jpg"):
    print "exists", processed_path
    return
 
 
  res = process('mkdir -p '+ processed_path)
  
  detector = cv2.AKAZE_create()
  detector.setThreshold(AKAZEThreshold)

  try: 
    r = requests.get(inpath)
    im = Image.open(StringIO(r.content))
    (width, height) = im.size
    swidth, sheight = width*0.16,height*0.16
    im.thumbnail( (swidth, sheight ), Image.ANTIALIAS)
  except:  
    print 'imge dl prob', inpath
    quit()
 

  img1 = np.array(im) 

  img1 = img1[:, :, ::-1].copy() 
  img1 = oc.clahergb(img1,cr=1.4,cg=1.4,cb=1.4)
  # img1 = cv2.imread(inpath)
  # small = cv2.resize(img1, (0,0), fx=0.16, fy=0.16) 
  
  cv2.imwrite(processed_path+"small.jpg", img1)
  cv2.imwrite("small.jpg", img1)
  ts=tnow() 
  kp1, desc1 = detector.detectAndCompute(img1, None)

  te=tnow()
  timetaken = te-ts
  print len(kp1), "kp", te-ts , "s"
  # ['__class__', '__delattr__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'angle', 'class_id', 'octave', 'pt', 'response', 'size']
  feats=''

  n1=0
  for nkp in range(len(kp1)):
    kp = kp1[nkp]
    float_array = [kp.pt[0], kp.pt[1], kp.size,kp.angle,  kp.response, kp.octave]
    keyp = struct.pack('6f', kp.pt[0], kp.pt[1], kp.size, kp.angle,  kp.response, kp.octave)
 
    g=keyp+desc1[nkp].tostring()
    ss=np.fromstring(g[-61:], dtype=np.uint8)
    feats += g
    n1+=1

  with open(processed_path+'feats.bin', 'wb') as f:
    f.write(feats) 

def imnum(nd):
    ix=nd%999;
    iy= int(math.floor(nd/999))
    if ix ==0 : 
     ix = 999
     iy -= 1
    return ix, iy


####################################################################
##    main                                                            
####################################################################
if __name__ == "__main__":

#  Config.FI = "http://10.0.0.111/F7T/Flight-Imagery/"
  Config.processed = "/home/oc/F5T/oc/Flight-Imagery/work/processed/"
  processed = "/home/oc/sdb1/processed/"
  AKAZEThreshold = 0.0005

  utc=arrow.utcnow()

  date = ['20151025']
  date = ['20150815']
  date = ['20150104']
  date = ['20150104']
  date = ['20140513']
  date = ['20150124']
  date = ['20140513']
  date = ["20170906"]
  FI = "http://127.0.0.1/sda1/FI4/"
  

  ts= tnow()
  step = 10
  mfa=['mf00','mf01','mf02','mf03']
  mfa=[ 'mf00','mf01','mf02','mf03' ]
  mfa=[ 'wf00', 'mf00','mf01','mf02','mf03' ]
  mfa=[ 'sw']

  # for nd in range(20,3640,step): 
  irange = range(1100,5902,step)


  for nd in irange: 
    for mf in mfa:
      ix,iy = imnum(nd)
   
      imuri = '/'+date[0]+'/'+mf+'/1'+"%02d"%iy+'D3200/DSC_'+"%04d"%ix+'.JPG'
      ts= oc.tnow()
      dncDSC2DB(imuri, fi = FI,processed = processed ,AKAZEThreshold=AKAZEThreshold, maxn=0, testdnc=1)
      te= oc.tnow()
      print 'dnc', imuri, "%0.3f"%(te-ts)
      
  te= tnow()
 
 
 # dncDSC2DB(imuri, fi = "http://10.0.0.111/F7T/Flight-Imagery/", processed = "/home/oc/F5T/oc/Flight-Imagery/work/processed/" ,AKAZEThreshold=0.0005, maxn=0, testdnc=1)
      










