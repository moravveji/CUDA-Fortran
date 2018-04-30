
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
                                  fmt_hdr="(a4,2(1x, a9))", &
                                  fmt_ascii="(i4,4(1x,f12.4))"
  character(len=9), parameter, dimension(3) :: hdr=(/ 'Num', 'T (ms)', 'BW (GB/s)' /)

  type(cudaEvent) :: start, finish
  real :: elapsed, bw

  real :: bw_offset(nmax), bw_stride(nmax), bw_stride_intent(nmax), bw_texture(nmax)

  type(cudaDeviceprop) :: prop
  integer, parameter :: handle = 1
  character(len=32) :: ascii
  logical, parameter :: write_ascii = .true.

  ! which device?
  ierr = cudaGetDeviceProperties(prop, 0)

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

  print*, '%%% Strided Access with Explicit Intent %%%'
  a_d = 0.0
  call stride_intent<<<n_blocks, block_size>>> (b_d, a_d, 1)
  write(*, fmt_hdr) hdr
  do k = 1, nmax
     a_d  = 0.0
     ierr = cudaEventRecord(start, 0)
     call stride_intent<<<n_blocks, block_size>>> (b_d, a_d, k)
     ierr = cudaEventRecord(finish, 0)
     ierr = cudaEventSynchronize(finish)
     ierr = cudaEventElapsedTime(elapsed, start, finish)
     bw   = 2 * nB / (elapsed * 1e6) 
     bw_stride_intent(k) = bw
     write(*,fmt_screen) k, elapsed, bw
  enddo

  print*, '%%% Strided Texture Access %%%'
  a_d = 0.0
  b_d = 0.0
  if (allocated(a_d)) then
    a_Tex => a_d
  else
    stop 'Error: a_d not allocate yet. Cannot associate ptr to it'
  endif 

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
     write(*, fmt_screen) k, elapsed, bw
  enddo

  ! wrap up
  if (write_ascii) then
     write(ascii, '(a,a)') trim(prop%name) // '.txt'
     open(unit=handle, file=trim(ascii), form='formatted', &
          status='replace', action='write')
     
     write(unit=handle, '(a4, 4(1x, a12))', advance='yes') 'k', 'misaligned', 'stride', &
                                                          'with_intent', 'texture'
     do k = 1, nmax
        write(unit=handle, fmt_ascii, advance='yes') k, bw_offset(k), & 
                                   bw_stride(k), bw_stride_intent(k), bw_texture(k)
     enddo
     close(unit=handle)
  endif
  nullify(a_Tex)
  deallocate(a_d, b_d)
  ierr = cudaEventDestroy(start)
  ierr = cudaEventDestroy(finish)

end program main_access
