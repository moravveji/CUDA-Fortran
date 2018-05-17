
program tex_vs_intent

  use cudafor
  use kernels

  implicit none

  integer, parameter :: n = 4 * 1024
  real, dimension(n) :: a, u
  real, dimension(n), target, device :: a_d, u_d
  integer, parameter :: n_iter = 10
  integer, parameter :: bs = 128, nb = n / bs ! block size & num. blocks
  integer :: iter, ierr
  real :: dt
  type(cudaEvent) :: start, finish

  a(:) = 1.23456
  a_d  = a
  u_d  = 0.0
  tex  => a_d
  call increment_tex<<<nb, bs>>> (u_d) 

  ierr = cudaEventCreate(start)
  ierr = cudaEventCreate(finish)

  ! Using Texture 
  ierr = cudaEventRecord(start, 0)
  do iter = 1, n_iter
     call increment_tex<<<nb, bs>>> (u_d)
     ierr = cudaDeviceSynchronize()
  enddo
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt, start, finish)
  dt   = dt / n_iter
  print '(a, f8.2)', 'Explicit texture memory: dt [mili.sec] = ', dt
  
  ! using intent(in) with device array
  call increment_intent<<<nb, bs>>> (a_d, u_d)

  ierr = cudaEventRecord(start, 0)
  do iter = 1, n_iter
     call increment_intent<<<nb, bs>>> (a_d, u_d)
     ierr = cudaDeviceSynchronize()
  enddo
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt, start, finish)
  dt   = dt / n_iter
  print '(a, f8.2)', 'intent(in) global memory: dt [mili.sec] = ', dt

  ! wrap up
  ierr = cudaEventDestroy(start)
  ierr = cudaEventDestroy(finish)

end program tex_vs_intent
