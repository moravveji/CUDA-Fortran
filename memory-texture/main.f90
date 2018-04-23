

program main

  use cudafor
  use kernels
  
  implicit none
  integer, parameter :: n = 1024
  integer, parameter :: nb = 4, bs = n / nb
  real, dimension(n), target, device :: zeros_d, ones_d
  real, dimension(n) :: ones
  integer :: ierr

  zeros_d = 0.0
  ones_d  = 0.0
  tex => zeros_d
  call add_tex<<<nb, bs>>> (ones_d)
  print*, 'first OK'
  ierr = cudaThreadSynchronize()
  print*, ierr, cudaSuccess
  call add_tex<<<nb, bs>>> (ones_d)
  print*, 'second OK'
  ones = ones_d
  print*, n !, sum(ones)

end program main
