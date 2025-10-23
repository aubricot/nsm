import os, argparse, pyvista as pv
def conv(inf, outf): pv.read(inf).save(outf)
def main():
 p=argparse.ArgumentParser(); p.add_argument('in_folder'); p.add_argument('out_folder')
 args=p.parse_args()
 if not os.path.exists(args.out_folder): os.makedirs(args.out_folder)
 for f in os.listdir(args.in_folder):
  if f.lower().endswith('.ply'):
   inf=os.path.join(args.in_folder, f); outf=os.path.join(args.out_folder, f[:-4]+'.vtk')
   print("Converting", inf, "to", outf); conv(inf,outf)
if __name__=='__main__': main()
