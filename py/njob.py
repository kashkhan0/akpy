import os, sys, json



wflist = [ [1600,1820],[2040,2280],[2790,3020],[3260,3410],[3750,4140],[4350,4630],[4860,5180],[5540,5720] ]
wflist = [[380,440],[650,720],[1080,1210],[1380,1470],[1780,1930],[2130,2230]]
wflist = [[2470,2570],[2860,2950],[3170,3310],[3600,3690],[3860,3970],[4320,4420],[4680,4850],[5280,5420],[5690,5930],[6490,6590],[6840,7010],[7640,7790],[8100,8330],[8450,8660],[8890,9200],[9290,9570],[9750,9980]]


#wflist = [[380,440,]]

date = "20171130"
tiledir = "/home/oc/sdb1/processed/20171118test19/"

zlist = [[17,6],[19,18],[21,20]]
fn ="1"


wfarr = []
mfarr = []
for wfdo in wflist:
  for n in range(wfdo[0], wfdo[1]):
    iy = int(n/999) 
    ix = n%999
    imuri = "/"+date+"/wf/1%02d"%iy+"D3200/DSC_0%03d"%ix+".JPG"
    wfarr.append(imuri)
    for mfn in range(7):
       mf = "mf%02d"%mfn
       imuri = "/"+date+"/"+mf +"/1%02d"%iy+"D3200/DSC_0%03d"%ix+".JPG"
       jj = {"imuri": imuri}
       mfarr.append(imuri)


wffn = "wfsall.txt"
with open(wffn, 'w') as f: f.write('')

for wf in wfarr: 
  js = {"imuri": wf}
  with open(wffn, 'a') as f: f.write(json.dumps(js)+'\n')

mffn = "mfsall.txt"
with open(mffn, 'w') as f: f.write('')

for mf in mfarr:
  js = {"imuri": mf}
  with open(mffn, 'a') as f: f.write(json.dumps(js)+'\n')


outmflist = "mflistall"+date+".txt"
with open(outmflist, 'w') as f: f.write('')
for wfdo in wflist:
  for n in range(wfdo[0], wfdo[1]):
     iy = int(n/999) 
     ix = n%999
     for mfn in range(7):
       mf = "mf%02d"%mfn
       imuri = "/"+date+"/"+mf +"/1%02d"%iy+"D3200/DSC_0%03d"%ix+".JPG"
       jj = {"imuri": imuri}
       with open(outmflist, 'a') as f: f.write(json.dumps(jj)+'\n')

print "wrote", outmflist



wfj=""

tc =0
for wfdo in wflist:  
  tc += wfdo[1]-wfdo[0] 
  tmpfn = "/tmp/qklistwf-"+fn+".txt"
  wfj+= "python scandata.py "+date+" wf  %0d"%wfdo[0]+" %0d"%wfdo[1]+ " 10 \n"
  wfj+= "./wfest -infn scanlist"+date+"wf-%0d"%wfdo[0]+"-%0d"%wfdo[1]+"s10o0.txt -redo 1 -mixdir  "+tiledir+"/  \n"
  wfj+= "python scandata.py "+date+" wf  %0d"%wfdo[0]+" %0d"%wfdo[1]+" \n"
  wfj+= "python scandata.py "+date+" wf  %0d"%wfdo[0]+" %0d"%wfdo[1]+ " 10 \n"
  wfj+= "./gcutwf -infn scanlist"+date+"wf-%0d"%wfdo[0]+"-%0d"%wfdo[1]+"s10o0.txt -redo 1 -vedir  /home/oc/sdb1/ve/ -step 1 -mixdir  "+tiledir+"/    -outfn "+tmpfn+" -imgsrc /home/oc/sdb1/ -zout 19 \n"
  wfj+= "./tileshrink -infn  "+tmpfn+"    -mixdir "+tiledir+"  -minzoom 18 \n"
  wfj+= "./gcutwf -infn scanlist"+date+"wf-%0d"%wfdo[0]+"-%0d"%wfdo[1]+"s10o0.txt -redo 1 -vedir  /home/oc/sdb1/ve/ -step 1 -mixdir  "+tiledir+"/  -outfn "+tmpfn+"  -imgsrc /home/oc/sdb1/ -zout 17 \n"
  wfj+= "./tileshrink -infn  "+tmpfn+"    -mixdir "+tiledir+"  -minzoom 6 \n"

mfj =""
for wfdo in wflist:
  tmpfn = "/tmp/qklistmf-"+fn+".txt"
  mfj+= "./mfestfast -infn scanlist"+date+"wf-%0d"%wfdo[0]+"-%0d"%wfdo[1]+".txt \n" 
  mfj+= "python scandata.py "+date+" mf  %0d"%wfdo[0]+" %0d"%wfdo[1] +" \n"
  mfj+= "./impcombo -infn scanlist"+date+"mf-%0d"%wfdo[0]+"-%0d"%wfdo[1]+".txt -vedir "+tiledir+" -zwork 19 -redo 1 \n"
  mfj+= "python scandata.py "+date+" mf  %0d"%wfdo[0]+" %0d"%wfdo[1] +"\n"
  for zz in zlist:
    gs = "17"
    seam = "1"
    if zz[0] < 20: 
       gs = "9"
       seam = "1"
    if zz[0] < 18: 
       gs = "3"
       seam = "0"
    mfj+= "./gcutwf -infn  scanlist"+date+"mf-%0d"%wfdo[0]+"-%0d"%wfdo[1]+".txt  -step 1 -mixdir  "+tiledir+"/ -vedir  /home/oc/sdb1/ve/  -outfn "+tmpfn+"  -imgsrc /home/oc/sdb1/ -zout "+str(zz[0])+" -redo 1 -gs "+gs+" -seam "+seam+" \n"
    mfj+= "./tileshrink -infn  "+tmpfn+"    -mixdir "+tiledir+"  -minzoom "+str(zz[1])+" \n"

with open("wfj.txt", "w") as fout: fout.write(wfj)
with open("mfj.txt", "w") as fout: fout.write(mfj)

print tc
