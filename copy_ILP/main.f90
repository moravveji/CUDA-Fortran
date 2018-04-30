
program main

  use cudafor
  use kerns

  implicit none

  integer, parameter :: MB = 4, KB = MB * 1024, B = KB * 1024, n = B / 4
  real, dimension(n), device :: src_d, targ_d
  real, dimension(n) :: src, targ

  integer, parameter :: bs = 256, nb = n / bs
  type(dim3) :: grid=dim3(nb, 1, 1), block=dim3(bs, 1, 1)

  type(cudaEvent) :: start, finish
  real :: dt, bw

  integer :: k, ierr, bytes_shmem
  integer, parameter :: ilp = bs
  real, parameter :: n_float = float(n)

  ! prepare
  print*, 'n_elements = ', n
  src  = 0.0
  targ = 0.0
  ierr = cudaEventCreate(start)
  ierr = cudaEventCreate(finish)
  bytes_shmem = sizeof(src_d(1)) * nb

  do k = 1, n
     src(k) = k / n_float
  enddo
  src_d = src

!  call copy_ilp<<<grid, block>>> (src_d, targ_d, ilp)
  call copy_ilp<<<1, bs>>> (src_d, targ_d, ilp)
  targ = targ_d
!  print*, src(1), src(n)
!  print*, targ(bs), targ(bs+1)
  print*, 'max error with ILP: ', maxval(abs(src - targ))

  ! Warm up
  call copy_global_mem<<<1, bs>>> (src_d, targ_d, nb)
  call copy_shared_mem<<<1, bs, bytes_shmem>>> (src_d, targ_d, nb)
  print*, 'Warm up: OK'
  ierr = cudaDeviceSynchronize()

  ! Now, copy by launching a single block, but the kernels goes over all other
  ! blocks internally over the global memory
  targ = 0.0
  targ_d = 0.0
  ierr = cudaEventRecord(start, 0)
  call copy_global_mem<<<1, bs>>> (src_d, targ_d, nb)
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt, start, finish)
  bw   = 4 * B / (dt * 1e6)  ! 4 is because kernel has four R/W operations 
  targ = targ_d
  print*, 'max error copy_global_mem: ', maxval(abs(src - targ))
  print*, 'BW [GB/sec] = ', bw

  ! Now, copy by launching single blocks, but with the intervention of shared
  ! memory
  targ = 0.0
  targ_d = 0.0
  ierr = cudaEventRecord(start, 0)
  call copy_shared_mem<<<1, bs, bytes_shmem>>> (src_d, targ_d, nb)
  ierr = cudaEventRecord(finish, 0)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt, start, finish)
  bw   = 4 * B / (dt * 1e6)  ! 4 is because kernel has four R/W operations
  targ = targ_d
  print*, 'max error copy_shared_mem: ', maxval(abs(src - targ))
  print*, 'BW [GB/sec] = ', bw

  ! wrap up
  ierr = cudaEventDestroy(start)
  ierr = cudaEventDestroy(finish)
end program main
