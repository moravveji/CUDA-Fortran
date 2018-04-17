
program main_access

  use cudafor

  use kern_misaligned
  use kern_strided  
  use kern_texture

  implicit none

  integer, parameter :: nMB = 4, nKB = 1024 * nMB, nB = 1024 * nKB, n = nB/4
  integer, parameter :: nmax = 32
  integer, parameter :: block_size = 256, n_blocks = n / block_size
  real, device, allocatable, target :: a_d(:), b_d(:)
  integer :: k, ierr
  character(len=24), parameter :: fmt_screen="(i4, 2(1x, f9.4))", &
                                  fmt_hdr="(a4,2(1x, a9))"
  character(len=9), parameter, dimension(3) :: hdr=(/ 'Num', 'T (ms)', 'BW (GB/s)' /)

  type(cudaEvent) :: start, finish
  real :: elapsed, bw

  real :: bw_offset(nmax), bw_stride(nmax), bw_texture(nmax)

  allocate(a_d(n*nmax), b_d(n), stat=ierr)
  if (ierr /= 0) stop 'Error: Allocate failed'

  ierr = cudaEventCreate(start)
  ierr = cudaEventCreate(finish)

  print*, ' %%% Offset Access %%%'
  ! warm up
  a_d = 0.0
  b_d = 0.0
  call offset<<<n_blocks, block_size>>> (b_d, a_d, 0)
  write(*, fmt_hdr) hdr

  ! measure bandwidth
  do k = 0, nmax-1
     b_d = 0.0
     ierr = cudaEventRecord(start, 0)
     call offset<<<n_blocks, block_size>>> (b_d, a_d, k)
     ierr = cudaEventRecord(finish, 0)
     ierr = cudaEventSynchronize(finish)
     ierr = cudaEventElapsedTime(elapsed, start, finish)
     bw   = 2 * nB / (elapsed * 1e6)
     bw_offset(k+1) = bw
     write(*, fmt_screen) k, elapsed, bw
  enddo
  print* 
  
  print*, '%%% Strided Access %%%'
  a_d = 0.0
  call stride<<<n_blocks, block_size>>> (b_d, a_d, 1)
  write(*, fmt_hdr) hdr
  do k = 1, nmax
     a_d  = 0.0
     ierr = cudaEventRecord(start, 0) 
     call stride<<<n_blocks, block_size>>> (b_d, a_d, k)
     ierr = cudaEventRecord(finish, 0)
     ierr = cudaEventSynchronize(finish)
     ierr = cudaEventElapsedTime(elapsed, start, finish)
     bw   = 2 * nB / (elapsed * 1e6)
     bw_stride(k) = bw
     write(*, fmt_screen) k, elapsed, bw
  enddo
  print*

  print*, '%%% Strided Texture Access %%%'
  a_d = 0.0
  if (allocated(a_d)) then
    a_Tex => a_d
  else
    stop 'Error: a_d not allocate yet. Cannot associate ptr to it'
  endif 
  print*, 'allocated:', allocated(a_Tex), 'associated:', associated(a_Tex)
  !stop 0

  call stride_texture<<<n_blocks, block_size>>> (b_d, 1)

  do k = 1, nmax
     a_d  = 0.0 
     ierr = cudaEventRecord(start, 0)
     call stride_texture<<<n_blocks, block_size>>> (b_d, k)
     ierr = cudaEventRecord(finish, 0)
     ierr = cudaEventSynchronize(finish)
     ierr = cudaEventElapsedTime(elapsed, start, finish)
     bw   = 2 * nB / (elapsed * 1e6)
     bw_texture(k) = bw
     print*, k, elapsed, bw
  enddo

  ! wrap up
  nullify(a_Tex)
  deallocate(a_d, b_d)
  ierr = cudaEventDestroy(start)
  ierr = cudaEventDestroy(finish)

end program main_access
