#import math
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import shiftnet_cuda

from torch.autograd import Variable
from torch.autograd import Function
import time

class ShiftFn(Function):
  @staticmethod
  def forward(ctx, src):
    dst = torch.cuda.FloatTensor(src.size())
    ret = shiftnet_cuda.moduloshift3x3_nchw(src, dst)
    assert ret == 1
    return dst

  @staticmethod
  def backward(ctx, grad_dst):
    grad_src = torch.cuda.FloatTensor(grad_dst.data.size())
    ret = shiftnet_cuda.moduloshift3x3bwd_nchw(grad_dst.data, grad_src)
    assert ret == 1
    return Variable(grad_src, requires_grad=grad_dst.requires_grad)

class Shift3x3(nn.Module):
  def __init__(self):
    super(Shift3x3, self).__init__()

  def forward(self, src):
    print("DEBUG: fwd:", type(src))
    return ShiftFn.apply(src)

class Shift3x3_cuda(nn.Module):
  def __init__(self):
    super(Shift3x3_cuda, self).__init__()

  def forward(self, src):
    return ShiftFn.apply(src)

if __name__ == "__main__":
  #import sys
  #sys.path.append("./")

  #import shift_module
  import numpy as np
  from torch.autograd import Variable

  pattern = np.arange(25).reshape(5,5)
  src_buf = np.zeros((16, 10, 5, 5)).astype(np.float32)
  for bnr in range(16):
    for ch in range(10):
      src_buf[bnr,ch,:,:] = pattern
  src = Variable(torch.from_numpy(src_buf).cuda(), requires_grad=True, volatile=False)
  print("DEBUG: src:", src.requires_grad)
  print(src.data.cpu().numpy()[0,:,:,:])

  shift = Shift3x3()

  out = shift.forward(src)
  print("DEBUG: out:", out.requires_grad)
  print(out.data.cpu().numpy()[0,:,:,:])

  sink = Variable(torch.ones(out.size()).cuda())
  out.backward(sink)
  print("DEBUG: src grad:")
  print(src.grad.data.cpu().numpy()[0,:,:,:])
