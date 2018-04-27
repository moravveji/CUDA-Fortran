import sys, os, glob
import numpy as np
import pylab as plt


def main():

  inputs = glob.glob('*.txt')
  n_inputs = len(inputs)
  if n_inputs == 0:
    raise IOError
    return -1

  dtp = {'names': ('k', 'misaligned', 'strided', 'intent', 'texture'), 
         'formats': ('i2', 'f4', 'f4', 'f4', 'f4')}
  for i_file in inputs:
    # start the canvas
    fig, ax = plt.subplots(1, figsize=(5, 4))

    data = np.loadtxt(fname=i_file, dtype=dtp, skiprows=1)

    ax.plot(data['k'], data['misaligned'], color='b', linestyle='solid', label='Misaligned')
    ax.plot(data['k'], data['strided'], color='g', linestyle='dashed', label='Stride')
    ax.plot(data['k'], data['intent'], color='g', linestyle='dotted', label='Stride:intent(in)')
    ax.plot(data['k'], data['texture'], color='k', linestyle='solid', label='Texture', linewidth=2)

    ax.legend(loc=1, frameon=False)
    ax.set_xlabel('Offset or Jump')
    ax.set_ylabel('Bandwidth (GB/sec)')
    ax.set_title(i_file.replace('.txt', ''))

    plt.tight_layout()
    fname = i_file.replace('.txt', '.png').replace(' ', '_')
    plt.savefig(fname, transparent=True)
    print 'Plot for "%s" created' % i_file
    plt.close()


if __name__ == '__main__':
  stat = main()
  sys.exit(stat)
