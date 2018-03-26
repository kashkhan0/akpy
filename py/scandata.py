import os, sys, json


datacache = "datacache/"
procdate  = "20170512"
mflist = ["wf"]
start=""
end=""
startn = 0
endn = 200000
step = 1
offset =0


if len(sys.argv) > 2:
  procdate = sys.argv[1]
  mflist = [sys.argv[2]]
  mfn = sys.argv[2]
  if sys.argv[2] == "mf":
    mflist = ["mf00","mf06", "mf01","mf02","mf03","mf04","mf05"]

if len(sys.argv) > 4:
  startn = int(sys.argv[3])
  endn= int(sys.argv[4])
  start = "-"+sys.argv[3]
  end = "-"+sys.argv[4]

if len(sys.argv) > 5:
   step = int(sys.argv[5])

if len(sys.argv) > 6:
   offset = int(sys.argv[6])

ststr = ""
if step !=1:
  ststr = "s"+str(step)+"o"+str(offset)
outfile= ("scanlist"+procdate+mfn+start+end+ststr+".txt").replace("/","")
outfail = ("scanfail"+procdate+mfn+start+end+ststr+".txt").replace("/","")
#outfile2= ("scanlistplane"+procdate+mfn+start+end+ststr+".txt").replace("/","")
#outfilemfest= ("scanlistmfest"+procdate+mfn+start+end+ststr+".txt").replace("/","")

with open(outfile, 'w') as fh:
  fh.write("") 
#with open(outfile2, 'w') as fh:
  fh.write("")
#with open(outfilemfest, 'w') as fh:
  fh.write("")
with open(outfail, 'w') as fh:
  fh.write("")


nlines = 0
nfail = 0
planelines = 0
mfestlines = 0
for n in range(12120):
  if n < startn or n > endn:
    continue
  sd = int(n/999)
  if (n+offset)%step != 0 :
    continue
  for mf in mflist:
    imuri = "/"+procdate+"/"+mf+"/1%02d"%sd+"D3200/DSC_%04d"%(n%999)+".JPG"
    akgeo = datacache +imuri+"/akazegeo.txt"
    failgeo = datacache +imuri+"/akfail.txt"
    if os.path.isfile(akgeo):
 #     print akgeo
     linejs = ""
     with open(akgeo) as fh:
       for line in fh:
         linejs = line.strip()
     if len(linejs) > 0: 
       with open(outfile, 'a') as fh:
         fh.write(linejs + "\n")
         nlines += 1
    if os.path.isfile(failgeo):
     linejs = ""
     with open(failgeo) as fh:
       for line in fh:
         linejs = line.strip()
     if len(linejs) > 0:
       with open(outfail, 'a') as fh:
         fh.write(linejs + "\n")
         nfail += 1



print "wrote", outfile, nlines, "lines", nfail, "nfail"



