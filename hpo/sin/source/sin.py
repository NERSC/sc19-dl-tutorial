import math
import argparse
import time

def main():
    # Parse command line args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-a", "--A", type=float, default=1.0)
    argparser.add_argument("-b", "--B", type=float, default=-1.0)
    argparser.add_argument("-c", "--C", type=float, default=1.0)
    argparser.add_argument("-d", "--D", type=float, default=-1.0)
    argparser.add_argument("-e", "--E", type=float, default=1.0)
    argparser.add_argument("-f", "--F", type=float, default=-1.0)
    argparser.add_argument("-g", "--G", type=float, default=1.0)
    args = argparser.parse_args()
    # Compute error
    start = time.time()
    err = 0.0
    for i in range(100):
        x = ((float(i)-50.0) / 50.0) * 3.1415926535
        val = args.A + args.B*pow(x,1) + args.C*pow(x,2) + args.D*pow(x,3)
        val = val    + args.E*pow(x,4) + args.F*pow(x,5) + args.G*pow(x,6)
        val = abs( math.sin(x) - val )
        val = val * val
        err = err + val
    stop = time.time()
    # Figure of merit scaled by run time
    fom = err*(stop-start)
    # Print the error as our FoM
    print("FoM: %e"%fom)

main()

