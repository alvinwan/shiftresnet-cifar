import sys
sys.path.append("./")

import shiftnet_cuda

import numpy as np
import torch
import torch.cuda

def main():
  pattern = np.arange(18 * 18).reshape(18, 18)
  src_buf = np.zeros((32, 64, 18, 18)).astype(np.float32)
  for bnr in range(32):
    for ch in range(64):
      src_buf[bnr,ch,:,:] = pattern

  x_hin = torch.zeros(32, 64, 18, 18).type(torch.FloatTensor)
  #x_hin[:,:,1:4,1:4] = 1.0
  x_hin.copy_(torch.from_numpy(src_buf))

  y_hin = torch.zeros(32, 64, 18, 18).type(torch.FloatTensor)

  x = x_hin.cuda()
  y = y_hin.cuda()

  #ret = shiftnet_cuda.moduloshift3x3_nchw(x, y)
  ret = shiftnet_cuda.moduloshiftgeneric_nchw(x, y, 7, 2, -1)
  assert ret == 1

  x_hout = x.cpu()
  y_hout = y.cpu()

  print(x_hout[0,0,:18,:18])
  for ch in range(9):
    print(y_hout[0,ch,:18,:18])

if __name__ == "__main__":
  main()
