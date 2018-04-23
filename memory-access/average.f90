
program average

  use cudafor
  use kern_averages
  
  implicit none
  
  ! work arrays
  real, device, target, dimension(0:n+1, 0:n+1) :: u_d
  real, device, dimension(n, n)  :: avr_d
  real, dimension(n, n)  :: avr
  real, device, dimension(0:n+1) :: x_d, y_d
!  real, dimension(0:n+1) :: x, y
  real, parameter :: x0=-1.0, x1=1.0, y0=-1, y1=1.0
  ! warm up arrays
  real, device, dimension(10, 10) :: w_u_d
  real, device, dimension(8, 8)   :: w_avr_d
  
  ! for performance
  integer, parameter :: bs = 256, nb = n/bs ! block size and num blocks
  type(cudaDeviceProp) :: prop
  character(len=32) :: dname
  type(cudaEvent) :: start, finish
  real :: dt_glob_4, dt_glob_8, dt_tex_4, dt_tex_8

  integer :: i, ierr

  ! initialize the recording events 
  ierr = cudaEventCreate(start)
  ierr = cudaEventCreate(finish)
  
  ! initiate the values
  u_d = 0.0
  avr_d = 0.0
  call init_1d<<<nb, bs>>> (x0, x1, x_d)
  call init_1d<<<nb, bs>>> (y0, y1, y_d)
  call init_2d<<<nb, bs>>> (x_d, y_d, u_d)

  ! which device?  
  ierr = cudaGetDeviceProperties(prop, 0)
  write(dname, '(a32)') prop%name
  print*, 'Using: ', trim(dname)

  ! warm up: call all four kernels
  w_u_d = 1.0
  w_avr_d = 0.0
  call avr_global_4<<<nb, bs>>> (w_u_d, w_avr_d)
  call avr_global_4<<<nb, bs>>> (w_u_d, w_avr_d)
  call avr_tex_4<<<nb, bs>>> (w_avr_d)
  call avr_tex_8<<<nb, bs>>> (w_avr_d)

  ! 4-cell average through global device memory
  ierr = cudaEventRecord(start, 0)
  call avr_global_4<<<nb, bs>>> (u_d, avr_d)
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt_glob_4, start, finish)

  ! 8-cell average through global device memory
  ierr = cudaEventRecord(start, 0)
  call avr_global_8<<<nb, bs>>> (u_d, avr_d)
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt_glob_8, start, finish)
print*, 'global succeeded'
  ! pointer associateion
  tex => u_d
  print*, 'tex => u_d'

  ! 4-cell average through texture device memory
  ierr = cudaEventRecord(start, 0)
  call avr_tex_4<<<nb, bs>>> (avr_d)
print*, 'avr_tex_4'
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt_tex_4, start, finish)

  ! 8-cell average through texture device memory
  ierr = cudaEventRecord(start, 0)
  call avr_tex_8<<<nb, bs>>> (avr_d)
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt_tex_8, start, finish)

  ! wrap up
!  nullify(tex)
  ierr = cudaEventDestroy(start)
  ierr = cudaEventDestroy(finish)  

end program average

