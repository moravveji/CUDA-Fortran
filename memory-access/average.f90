
program average

  use cudafor
  use kern_averages
  
  implicit none
  
  ! work arrays
  ! n is defined and used in the module kern_averages
  real, dimension(0:n+1, 0:n+1) :: u
  real, device, target, dimension(0:n+1, 0:n+1) :: u_d
  real, dimension(n, n) :: avr_4pt, avr_8pt
  real, device, dimension(n, n)  :: avr_d
  real, dimension(n, n)  :: avr
  real, device, dimension(0:n+1) :: x_d, y_d
!  real, dimension(0:n+1) :: x, y
  real, parameter :: x0=-1.0, x1=1.0, y0=-1, y1=1.0
  ! warm up arrays
  real, device, target, dimension(10, 10) :: w_u_d
  real, device, dimension(8, 8)   :: w_avr_d
  
  ! for performance
  integer, parameter :: bs = 256, nb = n/bs ! block size and num blocks
  type(cudaDeviceProp) :: prop
  character(len=32) :: dname
  type(cudaEvent) :: start, finish
  real :: dt_glob_4, dt_glob_8, dt_tex_4, dt_tex_8
  real :: bw_glob_4, bw_glob_8, bw_tex_4, bw_tex_8
  real :: er_glob_4, er_glob_8, er_tex_4, er_tex_8

  integer :: i, ierr, j, k
  character(len=32), parameter :: frmt = '(a24, 2(f8.3, 1x))'
  ! initialize the recording events 
  ierr = cudaEventCreate(start)
  ierr = cudaEventCreate(finish)
  
  ! initiate the values
  u_d = 0.0
  avr_d = 0.0
  call init_1d<<<nb, bs>>> (x0, x1, x_d)
  call init_1d<<<nb, bs>>> (y0, y1, y_d)
  call init_2d<<<nb, bs>>> (x_d, y_d, u_d)

  ! reference host matrices
  u = u_d
  do k = 1, n
     do j = 1, n
        avr_4pt(j, k) = (u(j-1, k) + u(j+1, k) + u(j, k-1) + u(j, k+1)) / 4.0
        avr_8pt(j, k) = (u(j-1, k) + u(j+1, k) + u(j, k-1) + u(j, k+1) + &
                         u(j-1, k-1) + u(j-1, k+1) + u(j+1, k-1) + u(j+1, k+1) ) / 8.0
     enddo 
  enddo

  ! which device?  
  ierr = cudaGetDeviceProperties(prop, 0)
  write(dname, '(a32)') prop%name
  print*, 'Using: ', trim(dname)

  ! warm up: call all four kernels
  w_u_d = 1.0
  w_avr_d = 0.0
  call avr_global_4<<<nb, bs>>> (w_u_d, w_avr_d)
  call avr_global_4<<<nb, bs>>> (w_u_d, w_avr_d)
  tex => w_u_d
  call avr_tex_4<<<nb, bs>>> (w_avr_d)
  call avr_tex_8<<<nb, bs>>> (w_avr_d)
  nullify(tex)

  ! 4-cell average through global device memory
  ierr = cudaEventRecord(start, 0)
  call avr_global_4<<<nb, bs>>> (u_d, avr_d)
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt_glob_4, start, finish)
  avr  = avr_d
  bw_glob_4 = get_bw(dt_glob_4) 
  er_glob_4 = maxval(abs(avr - avr_4pt))

  ! 8-cell average through global device memory
  ierr = cudaEventRecord(start, 0)
  call avr_global_8<<<nb, bs>>> (u_d, avr_d)
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt_glob_8, start, finish)
  avr  = avr_d
  bw_glob_8 = get_bw(dt_glob_8)
  er_glob_8 = maxval(abs(avr - avr_8pt))

  ! pointer associateion
  tex => u_d

  ! 4-cell average through texture device memory
  ierr = cudaEventRecord(start, 0)
  call avr_tex_4<<<nb, bs>>> (avr_d)
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt_tex_4, start, finish)
  avr  = avr_d
  bw_tex_4 = get_bw(dt_tex_4)
  er_tex_4 =  maxval(abs(avr - avr_4pt))

  ! 8-cell average through texture device memory
  ierr = cudaEventRecord(start, 0)
  call avr_tex_8<<<nb, bs>>> (avr_d)
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt_tex_8, start, finish)
  avr  = avr_d
  bw_tex_8 = get_bw(dt_tex_8)
  er_tex_8 = maxval(abs(avr - avr_8pt))

  ! write out a report
  write(*, frmt) '4-point global memory: ', bw_glob_4, er_glob_4
  write(*, frmt) '8-point global memory: ', bw_glob_8, er_glob_8
  write(*, frmt) '4-point texture memory:', bw_tex_4,  er_tex_4
  write(*, frmt) '8-point texture memory:', bw_tex_8,  er_tex_8

  ! wrap up
  nullify(tex)
  ierr = cudaEventDestroy(start)
  ierr = cudaEventDestroy(finish)  


  contains

  !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  real function get_bw(dt) result(bw)
    real, intent(in) :: dt
    bw = 4 * (n**2 + (n+2)**2) / (dt * 1e6)
  end function get_bw
  !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end program average

