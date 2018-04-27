

program main

  use cudafor
  use kernels
  
  implicit none
  integer, parameter :: n = 256
  integer, parameter :: nb = 1, bs = n / nb
  integer :: bytes
  real, dimension(n), target, device :: zeros_d
  real, dimension(n), device :: ones_d
  real, dimension(n) :: ones
  integer :: ierr

  real, target, device :: dummy_d(n)
  real, device :: dummy_returned(n)
  real :: dummy_h(n)

  dummy_d(:) = 123.456
  dummy => dummy_d
  call get_texture<<<nb, bs>>> (dummy_returned)
  dummy_h = dummy_returned  ! D2H
  print*, dummy_h(1)

  bytes   = sizeof(ones_d(1)) * n
  zeros_d = 0.0
  ones_d  = 0.0
  tex => zeros_d
  call increment_tex<<<nb, bs>>> (ones_d)
  ierr = cudaThreadSynchronize()
  if (ierr /= cudaSuccess) then
     print*, 'Error: kernel launch of cudaThreadSynchronize() failed; ierr=', ierr
     print('(3(a10,i6))'), 'nb = ', nb, 'bs = ', bs, 'n = ', n
     print*, cudaGetErrorString(ierr)
     call exit(ierr)
  else
     print*, 'Kernel successfully launched and completed'
  endif
  ones = ones_d

end program main
