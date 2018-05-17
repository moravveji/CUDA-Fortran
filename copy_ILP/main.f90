
program main

  use cudafor
  use kerns

  implicit none

  integer, parameter :: MB = 6, KB = MB * 1024, B = KB * 1024, n = B / 4
  real, dimension(n), device :: src_d, targ_d
  real, dimension(n) :: src, targ

  integer, parameter :: bs = 1024, nb = n / bs
  type(dim3) :: grid=dim3(nb, 1, 1), block=dim3(bs, 1, 1)

  type(cudaEvent) :: start, finish
  real :: dt, bw

  integer(kind=cuda_stream_kind) :: str1, str2, str3, str4, str5, str6, str7, str8

  integer :: k, ierr, bytes_shmem
  integer, parameter :: ilp = bs
  real, parameter :: n_float = float(n)

  real, dimension(n), device :: s1_d, s2_d, s3_d, s4_d, s5_d, s6_d, s7_d, s8_d, &
                                t1_d, t2_d, t3_d, t4_d, t5_d, t6_d, t7_d, t8_d
 
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
  print*, 'Elapsed time [mili.sec] = ', dt
  print*, 'BW [GB/sec] = ', bw

  ! Now, copying 8 source arrays to target arrays through 8 different streams
  s1_d = 1.0; s2_d = 2.0; s3_d = 3.0; s4_d = 4.0
  s5_d = 5.0; s6_d = 6.0; s7_d = 7.0; s8_d = 8.0

  ierr = cudaStreamCreate(str1)
  ierr = cudaStreamCreate(str2)
  ierr = cudaStreamCreate(str3)
  ierr = cudaStreamCreate(str4)
  ierr = cudaStreamCreate(str5)
  ierr = cudaStreamCreate(str6)
  ierr = cudaStreamCreate(str7)
  ierr = cudaStreamCreate(str8)

  ierr = cudaEventRecord(start, str1)
  call copy_shared_mem<<<1, bs, bytes_shmem, str1>>> (s1_d, t1_d, nb)
  call copy_shared_mem<<<1, bs, bytes_shmem, str2>>> (s2_d, t2_d, nb)
  call copy_shared_mem<<<1, bs, bytes_shmem, str3>>> (s3_d, t3_d, nb)
  call copy_shared_mem<<<1, bs, bytes_shmem, str4>>> (s4_d, t4_d, nb)
  call copy_shared_mem<<<1, bs, bytes_shmem, str5>>> (s5_d, t5_d, nb)
  call copy_shared_mem<<<1, bs, bytes_shmem, str6>>> (s6_d, t6_d, nb)
  call copy_shared_mem<<<1, bs, bytes_shmem, str7>>> (s7_d, t7_d, nb)
  call copy_shared_mem<<<1, bs, bytes_shmem, str8>>> (s8_d, t8_d, nb)
  ierr = cudaEventRecord(finish, str8)
  ierr = cudaEventSynchronize(finish)
  ierr = cudaEventElapsedTime(dt, start, finish)
  bw   = 8 * 4 * B / (dt * 1e6)  ! 8 is for 8 independent streams
  print*, 'Result for 8 copy streams'
  print*, 'Elapsed time [mili.sec] = ', dt
  print*, 'BW [GB/sec] = ', bw
  ! wrap up
  ierr = cudaEventDestroy(start)
  ierr = cudaEventDestroy(finish)
end program main
